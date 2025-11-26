import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import networkx as nx
import warnings
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import time
import pickle
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
import re

warnings.filterwarnings("ignore")

# ============================================================================
# OPENAI SETUP
# ============================================================================

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print(
        "⚠️ Warning: OPENAI_API_KEY not found in .env file. Topic labels will use fallback method."
    )
    USE_OPENAI = False
else:
    USE_OPENAI = True

# System prompt for topic labeling
TOPIC_LABELING_SYSTEM_PROMPT = """You are an expert at analyzing social media discussions and creating clear, descriptive topic labels. 
Your task is to generate concise, human-readable topic names based on keywords and sample comments from TikTok.

Guidelines:
- Create descriptive labels that capture the essence of what people are discussing
- Use 5-10 words maximum
- Be specific rather than generic
- Focus on the main theme or issue being discussed
- Use natural, conversational language that anyone can understand
- Avoid using the exact keywords in sequence; instead create a flowing description"""

# ============================================================================
# LOAD YOUR DATA (Assuming you have monthly_data.pkl from previous steps)
# ============================================================================

print("Loading monthly data...")
with open("monthly_data.pkl", "rb") as f:
    monthly_data = pickle.load(f)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================================
# STEP 1: RUN TOPIC MODELING WITH HDBSCAN
# ============================================================================

monthly_topics = {}
monthly_topic_info = {}
monthly_topic_representations = {}
monthly_outliers = {}

print("\n" + "=" * 60)
print("RUNNING TOPIC MODELING WITH HDBSCAN")
print("=" * 60)


def filter_low_information_texts(texts, min_tokens=3):
    """
    Remove comments that lack enough informative tokens to survive vectorization.
    This reduces empty-vocabulary failures caused by tiny batches, repetitive
    text, or stopword-heavy content by enforcing a floor on non-stopword tokens
    per comment.
    """

    filtered = []
    dropped = 0
    for text in texts:
        raw_text = str(text)
        tokens = [
            tok
            for tok in re.findall(r"\b\w+\b", raw_text.lower())
            if tok not in ENGLISH_STOP_WORDS
        ]
        if len(tokens) >= min_tokens:
            # Keep the original comment content so embeddings retain context; we only
            # use the token count to screen out near-empty, stopword-only texts.
            filtered.append(raw_text)
        else:
            dropped += 1
    return filtered, dropped

for month_str in sorted(monthly_data.keys()):
    print(f"\nProcessing month: {month_str}")
    month_df = monthly_data[month_str]
    texts = month_df["processed_text"].tolist()

    texts, dropped = filter_low_information_texts(texts)
    if dropped:
        print(
            f"  ⚠️ Filtered {dropped} low-information comments (<{3} non-stopword tokens)"
        )

    if len(texts) < 10:
        print(f"  ⚠️ Skipping {month_str}: Only {len(texts)} comments")
        continue

    n_docs = len(texts)

    # Configure HDBSCAN with adaptive parameters
    hdbscan_model = HDBSCAN(
        # min_cluster_size defines the smallest grouping HDBSCAN will accept as a
        # topic; higher values merge nearby points so you get fewer, broader topics,
        # while lower values allow more granular topics at the risk of extra noise.
        # Bumping this upward reduces the total topic count per month.
        min_cluster_size=max(60, min(160, int(n_docs * 0.05))),
        # min_samples sets how strictly HDBSCAN distinguishes dense clusters from
        # noise; raising it yields sturdier clusters but more outliers, whereas
        # lowering it keeps more points in clusters but can blur topic boundaries.
        # Increasing this in tandem with min_cluster_size prunes fragile clusters so
        # the surviving topics cover more comments each.
        min_samples=min(30, max(10, int(n_docs * 0.01))),
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.1,
        prediction_data=True,
    )

    # Configure UMAP
    umap_model = UMAP(
        n_neighbors=min(35, max(15, int(n_docs * 0.01))),
        n_components=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    # Configure vectorizer
    # The vectorizer controls how raw text is converted into token counts that feed the
    # topic model. Tuning the document frequency thresholds helps balance vocabulary
    # richness against noise: lower values keep rare but meaningful terms, while higher
    # values filter out infrequent words that can fragment topics. Adjusting the
    # maximum feature count and n-gram range influences how granularly phrases are
    # captured, which in turn affects how cohesive and distinguishable the resulting
    # topics become. Raising max_features (e.g., from 10k to 15k) allows a larger
    # vocabulary that can surface finer distinctions between topics but may also add
    # sparsity and noise if the extra terms are not informative.

    def derive_vectorizer_thresholds(texts, n_docs):
        """
        Dynamically pick min_df/max_df that keep at least one token after pruning by
        probing term frequencies and relaxing thresholds as needed.
        """

        # Start with the adaptive mins used previously, but be willing to relax them if
        # the vocabulary would be emptied by pruning.
        if n_docs < 100:
            base_min_df = 2
        elif n_docs < 500:
            base_min_df = 3
        elif n_docs < 1000:
            base_min_df = 5
        else:
            base_min_df = 10

        max_df_ratio = 0.95

        # Probe the raw token document frequencies to avoid choosing a min_df that
        # would eliminate every term (e.g., when no token appears base_min_df times).
        probe_vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            max_features=10000,
            stop_words="english",
        )

        probe_vectorizer.fit(texts)
        probe_matrix = probe_vectorizer.transform(texts)
        doc_freqs = np.asarray(probe_matrix.astype(bool).sum(axis=0)).ravel()

        if doc_freqs.size == 0:
            raise ValueError("Empty vocabulary after probing")

        max_doc_freq = int(doc_freqs.max())

        min_df_value = min(base_min_df, max_doc_freq)
        min_df_value = max(min_df_value, 1)

        max_df_docs = int(np.floor(max_df_ratio * n_docs))
        max_df_docs = max(max_df_docs, min_df_value)

        fallback_used = False

        while True:
            tester = CountVectorizer(
                ngram_range=(1, 1),
                min_df=min_df_value,
                max_df=max_df_docs,
                max_features=10000,
                stop_words="english",
            )

            try:
                tester.fit(texts)
                vocab_size = len(tester.vocabulary_)
                if vocab_size == 0:
                    raise ValueError("Empty vocabulary after pruning")
                break
            except ValueError:
                fallback_used = True

                if min_df_value > 1:
                    min_df_value = max(1, min_df_value // 2)
                elif max_df_docs < n_docs:
                    max_df_docs = min(n_docs, max_df_docs + max(1, max_df_docs // 2))
                else:
                    # Cannot relax further; bail out so the caller can try the full
                    # relaxed fallback later.
                    raise

        log_msg = (
            f"    Vectorizer thresholds -> min_df: {min_df_value} docs, "
            f"max_df: {max_df_docs} docs (derived from ratio {max_df_ratio})"
        )
        if fallback_used:
            log_msg += " [adjusted to retain vocabulary]"
        print(log_msg)

        vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            min_df=min_df_value,
            max_df=max_df_docs,
            max_features=10000,
            stop_words="english",
        )
        return vectorizer

    try:
        vectorizer_model = derive_vectorizer_thresholds(texts, n_docs)
    except ValueError as e:
        print(f"  ❌ Error deriving vectorizer thresholds: {e}")
        continue

    # Larger values merge narrow clusters so fewer, broader topics emerge; smaller values
    # allow more granular topics at the risk of over-fragmentation. Raising these bounds
    # favors a compact set of high-coverage topics instead of many tiny ones.
    min_topic_size_value = max(60, min(170, int(n_docs * 0.06)))

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size_value,
        # Constrain the per-month topic set toward a compact range (roughly 20–25)
        # to emphasize the most dominant themes while discouraging excessive
        # fragmentation.
        nr_topics=22,
        calculate_probabilities=False,
        verbose=False,
    )

    try:
        topics, probs = topic_model.fit_transform(texts)
    except ValueError as e:
        if "After pruning, no terms remain" in str(e):
            print("  ⚠️ Retrying with relaxed vectorizer thresholds (min_df=1, max_df=1.0)")

            relaxed_vectorizer = CountVectorizer(
                ngram_range=(1, 1),
                min_df=1,
                max_df=1.0,
                max_features=10000,
                stop_words="english",
            )

            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=relaxed_vectorizer,
                min_topic_size=min_topic_size_value,
                nr_topics=22,
                calculate_probabilities=False,
                verbose=False,
            )

            topics, probs = topic_model.fit_transform(texts)
        else:
            raise
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        continue

    topic_info = topic_model.get_topic_info()

    n_outliers = (np.array(topics) == -1).sum()
    outlier_percentage = (n_outliers / len(topics)) * 100

    monthly_outliers[month_str] = {
        "count": n_outliers,
        "percentage": outlier_percentage,
        "outlier_texts": [texts[i] for i, t in enumerate(topics) if t == -1][:10],
    }

    # Get topic representations
    topic_representations = {}
    for topic_id in topic_info["Topic"].unique():
        if topic_id != -1:
            topic_words = topic_model.get_topic(topic_id)
            topic_representations[topic_id] = {
                "words": [word for word, score in topic_words[:10]],
                "scores": [score for word, score in topic_words[:10]],
            }

    monthly_topics[month_str] = {
        "model": topic_model,
        "topics": topics,
        "documents": texts,
    }
    monthly_topic_info[month_str] = topic_info
    monthly_topic_representations[month_str] = topic_representations

    n_topics = len(topic_info) - 1
    print(f"  ✓ Found {n_topics} topics")
    print(f"  ✓ Outliers: {n_outliers} ({outlier_percentage:.1f}%)")

# ============================================================================
# STEP 2: CALCULATE TOPIC EVOLUTION
# ============================================================================


def calculate_topic_similarity(topics1, topics2):
    similarity_matrix = []
    for t1_id, t1_repr in topics1.items():
        row = []
        for t2_id, t2_repr in topics2.items():
            all_words = set(t1_repr["words"]) | set(t2_repr["words"])
            if len(all_words) == 0:
                row.append(0.0)
                continue

            vec1 = np.zeros(len(all_words))
            vec2 = np.zeros(len(all_words))
            word_list = list(all_words)

            for idx, word in enumerate(word_list):
                if word in t1_repr["words"]:
                    word_idx = t1_repr["words"].index(word)
                    vec1[idx] = t1_repr["scores"][word_idx]
                if word in t2_repr["words"]:
                    word_idx = t2_repr["words"].index(word)
                    vec2[idx] = t2_repr["scores"][word_idx]

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                similarity = 0.0
            row.append(similarity)
        similarity_matrix.append(row)
    return np.array(similarity_matrix)


print("\n" + "=" * 60)
print("CALCULATING TOPIC EVOLUTION")
print("=" * 60)

topic_evolution = {}  # Stores month-to-month topic linkages that drive split/merge detection
sorted_months = sorted(monthly_topic_representations.keys())

for i in range(len(sorted_months) - 1):
    month1 = sorted_months[i]
    month2 = sorted_months[i + 1]

    print(f"Analyzing {month1} → {month2}")

    topics1 = monthly_topic_representations[month1]
    topics2 = monthly_topic_representations[month2]

    if not topics1 or not topics2:
        continue

    similarity_matrix = calculate_topic_similarity(topics1, topics2)

    connections = []
    STRONG_CONNECTION_THRESHOLD = 0.28

    for t1_idx, t1_id in enumerate(topics1.keys()):
        for t2_idx, t2_id in enumerate(topics2.keys()):
            sim = similarity_matrix[t1_idx, t2_idx]
            if sim > 0:
                connections.append(
                    {
                        "from_topic": t1_id,
                        "to_topic": t2_id,
                        "similarity": sim,
                        "strong_connection": sim >= STRONG_CONNECTION_THRESHOLD,
                    }
                )

    connections = sorted(connections, key=lambda x: x["similarity"], reverse=True)
    topic_evolution[f"{month1}->{month2}"] = {
        "similarity_matrix": similarity_matrix,
        "connections": connections,
        "from_month": month1,
        "to_month": month2,
    }

# ============================================================================
# STEP 3: BUILD TOPIC NETWORK WITH SPLITS/MERGES
# ============================================================================


class ImprovedTopicEvolutionNetwork:
    def __init__(
        self,
        monthly_representations,
        topic_evolution,
        similarity_threshold=0.5,
        min_branch_length=2,
    ):
        self.monthly_representations = monthly_representations
        self.topic_evolution = topic_evolution
        # similarity_threshold: minimum edge similarity for a topic to continue; lowering it
        # connects loosely related topics and increases merges/splits, while raising it
        # enforces stricter continuity and yields cleaner but fewer transitions.
        self.similarity_threshold = similarity_threshold
        # min_branch_length: shortest allowed offshoot; smaller values keep short-lived
        # branches visible (more splits), larger values suppress brief detours for a
        # cleaner mainline.
        self.min_branch_length = min_branch_length
        self.graph = nx.DiGraph()
        self.chains = []

    def build_evolution_graph(self):
        sorted_months = sorted(self.monthly_representations.keys())

        for month in sorted_months:
            for topic_id in self.monthly_representations[month].keys():
                node_id = f"{month}_{topic_id}"
                self.graph.add_node(node_id, month=month, topic_id=topic_id)

        for evolution_key, evolution_data in self.topic_evolution.items():
            from_month, to_month = evolution_key.split("->")
            for conn in evolution_data["connections"]:
                if conn["strong_connection"]:
                    from_node = f"{from_month}_{conn['from_topic']}"
                    to_node = f"{to_month}_{conn['to_topic']}"
                    self.graph.add_edge(
                        from_node, to_node, similarity=conn["similarity"]
                    )
        return self.graph

    def _trace_path_length(self, start_node, visited=None):
        if visited is None:
            visited = set()
        if start_node in visited:
            return 0
        visited.add(start_node)
        successors = list(self.graph.successors(start_node))
        if not successors:
            return 1
        max_length = 0
        for successor in successors:
            length = 1 + self._trace_path_length(successor, visited.copy())
            max_length = max(max_length, length)
        return max_length

    def build_filtered_chains(self):
        sorted_months = sorted(self.monthly_representations.keys())
        visited = set()
        chain_id = 0

        first_month = sorted_months[0]
        starting_nodes = [
            n
            for n in self.graph.nodes()
            if self.graph.in_degree(n) == 0 or n.startswith(f"{first_month}_")
        ]

        for start_node in starting_nodes:
            if start_node in visited:
                continue
            chain = self._build_filtered_chain(start_node, visited)
            if chain["nodes"]:
                chain["chain_id"] = f"Chain_{chain_id}"
                self.chains.append(chain)
                chain_id += 1

        return self.chains

    def _build_filtered_chain(self, start_node, visited):
        chain = {
            "nodes": [],
            "branches": [],
            "is_split": False,
            "is_merge": False,
            "chain_id": None,
        }

        current = start_node
        while current and current not in visited:
            visited.add(current)
            month, topic_id = current.split("_")

            node_data = {"month": month, "topic_id": int(topic_id), "node_id": current}

            if month in self.monthly_representations:
                if int(topic_id) in self.monthly_representations[month]:
                    words = self.monthly_representations[month][int(topic_id)]["words"][
                        :5
                    ]
                    node_data["words"] = words

            chain["nodes"].append(node_data)

            predecessors = list(self.graph.predecessors(current))
            if len(predecessors) > 1:
                chain["is_merge"] = True

            successors = list(self.graph.successors(current))

            if len(successors) == 0:
                break
            elif len(successors) == 1:
                current = successors[0]
            else:
                branch_lengths = []
                for successor in successors:
                    length = self._trace_path_length(successor, visited.copy())
                    branch_lengths.append((successor, length))

                substantial_branches = [
                    (s, l) for s, l in branch_lengths if l >= self.min_branch_length
                ]

                if len(substantial_branches) <= 1:
                    if substantial_branches:
                        current = substantial_branches[0][0]
                    else:
                        current = max(branch_lengths, key=lambda x: x[1])[0]
                else:
                    chain["is_split"] = True
                    best_successor = None
                    best_similarity = -1

                    for successor, length in substantial_branches:
                        edge_data = self.graph.get_edge_data(current, successor)
                        if (
                            edge_data
                            and edge_data.get("similarity", 0) > best_similarity
                        ):
                            best_similarity = edge_data["similarity"]
                            best_successor = successor

                    if best_successor:
                        current = best_successor
                        for successor, length in substantial_branches:
                            if successor != best_successor and successor not in visited:
                                branch = self._build_filtered_chain(successor, visited)
                                if branch["nodes"]:
                                    chain["branches"].append(branch)
                    else:
                        break
        return chain

    def calculate_chain_longevity(self, chain):
        if not chain["nodes"]:
            return 0
        main_longevity = len(chain["nodes"])
        if chain["branches"]:
            branch_longevities = [
                self.calculate_chain_longevity(branch) for branch in chain["branches"]
            ]
            return (
                main_longevity + max(branch_longevities)
                if branch_longevities
                else main_longevity
            )
        return main_longevity


# Build network
print("\n" + "=" * 60)
print("BUILDING TOPIC EVOLUTION NETWORK")
print("=" * 60)

network = ImprovedTopicEvolutionNetwork(
    monthly_representations=monthly_topic_representations,
    topic_evolution=topic_evolution,
    similarity_threshold=0.28,
    min_branch_length=1,
)

graph = network.build_evolution_graph()
print(
    f"✓ Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
)

filtered_chains = network.build_filtered_chains()
print(f"✓ Created {len(filtered_chains)} chains")

# ============================================================================
# STEP 4: ENHANCE WITH OPENAI (if available)
# ============================================================================


def call_openai_for_topic_label(keywords, sample_comments, max_retries=3):
    if not USE_OPENAI:
        return f"{', '.join(keywords[:3])} discussion"

    user_prompt = f"""Based on these topic keywords and sample comments, create a clear, descriptive topic label.

Topic Keywords: {', '.join(keywords[:5])}

Sample Comments from this topic:
"""

    for i, comment in enumerate(sample_comments[:5], 1):
        comment_preview = comment[:170] + "..." if len(comment) > 150 else comment
        user_prompt += f'{i}. "{comment_preview}"\n'

    user_prompt += "\nGenerate a descriptive topic label (5-10 words) that captures what users are discussing:"

    for attempt in range(max_retries):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(
                api_key=api_key,
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": TOPIC_LABELING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=50,
            )
            topic_label = (
                response.choices[0].message.content.strip().strip('"').strip("'")
            )
            return topic_label
        except Exception as e:
            print(f"  ⚠️ OpenAI error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return f"{', '.join(keywords[:3])} discussion"


def get_top_comments_for_topic(topic_assignments, documents, topic_id, n_comments=5):
    topic_indices = [i for i, t in enumerate(topic_assignments) if t == topic_id]
    if not topic_indices:
        return []
    topic_comments = [documents[i] for i in topic_indices[:n_comments]]
    return topic_comments


# Enhance representations
print("\n" + "=" * 60)
print("GENERATING HUMAN-READABLE LABELS")
print("=" * 60)

enhanced_monthly_representations = {}

for month in sorted(monthly_topic_representations.keys()):
    print(f"Processing {month}...")
    enhanced_monthly_representations[month] = {}

    if month not in monthly_topics:
        continue

    topic_assignments = monthly_topics[month]["topics"]
    documents = monthly_topics[month]["documents"]

    for topic_id, topic_data in monthly_topic_representations[month].items():
        enhanced_monthly_representations[month][topic_id] = topic_data.copy()
        keywords = topic_data["words"][:5]
        sample_comments = get_top_comments_for_topic(
            topic_assignments, documents, topic_id, n_comments=5
        )

        human_label = call_openai_for_topic_label(keywords, sample_comments)
        enhanced_monthly_representations[month][topic_id]["human_label"] = human_label
        enhanced_monthly_representations[month][topic_id][
            "original_keywords"
        ] = keywords

        if USE_OPENAI:
            time.sleep(0.5)  # Rate limiting

# ============================================================================
# STEP 5: CREATE FINAL VISUALIZATION (WITH YOUR IMPROVEMENTS)
# ============================================================================
# Graph linking thresholds and labeling knobs
# EDGE_SIM_PLOT_THRESHOLD: Controls which moderate-strength edges are drawn between nodes.
#   Impact on nodes: Lower values densify the network around nodes, potentially crowding them; higher values declutter but can isolate nodes.
#   Impact on merges/splits: Lower values reveal more candidate cross-month links, so splits/merges may appear more frequently; higher values hide borderline links.
#   Impact on topic size/noise: Lower values can surface noisy or coincidental similarities; higher values emphasize only stronger relationships.
#   Example: EDGE_SIM_PLOT_THRESHOLD = 0.5 would draw only edges with similarity ≥0.5, yielding a sparser, more conservative linkage set. Dropping it to 0.3 keeps weaker ties visible, promoting interconnections.
EDGE_SIM_PLOT_THRESHOLD = 0.30

# TOPIC_EMB_SIM_THRESHOLD: Embedding similarity required to treat topics as related when chaining.
#   Impact on nodes: Lower values keep more nodes connected across months; higher values leave more isolated nodes.
#   Impact on merges/splits: Lower values encourage merges and longer chains; higher values demand tighter semantic alignment, reducing split/merge events.
#   Impact on topic size/noise: Lower values may mix nearby but distinct themes (more noise); higher values keep topics purer but may fragment them.
#   Example: TOPIC_EMB_SIM_THRESHOLD = 0.50 would only link topics with cosine similarity ≥0.5, leading to fewer, stricter continuations, while 0.33 favors longer-lived chains.
TOPIC_EMB_SIM_THRESHOLD = 0.33

# DOC_TO_PREV_TOPIC_THRESHOLD: Minimum document overlap to continue a topic lineage.
#   Impact on nodes: Lower values keep nodes active even with modest shared comments; higher values prune nodes that do not share enough documents.
#   Impact on merges/splits: Lower values increase chances of continuations/merges; higher values can terminate branches sooner, reducing splits.
#   Impact on topic size/noise: Lower values risk carrying forward noisy or weakly related content; higher values ensure continuity reflects substantial audience overlap.
#   Example: DOC_TO_PREV_TOPIC_THRESHOLD = 0.20 would let topics persist with only 20% overlap, extending chains but with looser cohesion; 0.36 (current) prefers stronger overlap.
DOC_TO_PREV_TOPIC_THRESHOLD = 0.36

# BRIDGING_THRESHOLD: Similarity cutoff for strong “bridge” edges.
#   Impact on nodes: Lower values mark more edges as strong, visually reinforcing connectivity; higher values highlight only the most confident ties.
#   Impact on merges/splits: Lower values can signal more opportunities for merges; higher values restrict visible bridges, focusing on solid continuations.
#   Impact on topic size/noise: Lower values may admit noisy bridges; higher values keep only robust relationships, possibly shortening chains.
#   Example: BRIDGING_THRESHOLD = 0.25 would classify many moderate edges as strong, thickening the network and inviting more merges.
BRIDGING_THRESHOLD = 0.34

# TOP_TERM_COUNT: Number of top words retained per topic for labeling.
#   Impact on nodes: More terms add descriptive nuance to each node’s tooltip/label; fewer keep labels concise.
#   Impact on merges/splits: Does not alter structure directly but richer terms can make split/merge interpretations clearer or noisier.
#   Impact on topic size/noise: Higher counts may introduce peripheral terms (noise) into labels; lower counts focus on the core theme.
#   Example: TOP_TERM_COUNT = 15 would show 15 key words per topic label, offering more detail but risking clutter.
TOP_TERM_COUNT = 10

# EPHEMERAL_DOC_COUNT: Counts at or below this are treated as fleeting topics.
#   Impact on nodes: Lower values keep more tiny nodes visible; higher values suppress or flag very small topics.
#   Impact on merges/splits: Higher values may hide small branches that could have split/merge signals; lower values will show them, possibly increasing perceived splits.
#   Impact on topic size/noise: Lower thresholds retain noisy, low-volume topics; higher thresholds emphasize substantial topics and reduce speckle noise.
#   Example: EPHEMERAL_DOC_COUNT = 20 would treat topics with ≤20 comments as ephemeral, reducing their prominence or inclusion.
EPHEMERAL_DOC_COUNT = 6
ROW_SPACING = 1.25
N_COMMENTS_FOR_BOLD = 1000  # Topics exceeding this total comment count render labels in bold to highlight highly discussed themes.
COUNT_NODE_SIZE_SCALE = 1.25  # Scales the size of nodes in the count-only overlay to give text more room.
COUNT_NODE_FONT_SIZE = 6  # Font size for centered comment totals inside overlay nodes.


def create_clean_evolution_visualization_with_labels(
    network, chains, monthly_representations
):
    def calculate_layout(chains):
        layout = {}
        y_position = 0
        split_info = []
        lineage_counter = defaultdict(int)
        suppressed_lineages = set()
        lineage_parent = {}

        def prune_short_branches(part):
            retained = []
            for branch in part["branches"]:
                prune_short_branches(branch)
                if network.calculate_chain_longevity(branch) >= 2:
                    retained.append(branch)
            part["branches"] = retained

        for chain in chains:
            prune_short_branches(chain)
            if network.calculate_chain_longevity(chain) < 2:
                continue

            def process_chain_part(
                part,
                y_pos,
                parent_end=None,
                is_branch=False,
                branch_index=0,
                lineage_id=None,
                parent_doc_count=None,
            ):
                nonlocal y_position

                branch_inherits_parent_label = False

                for i, node in enumerate(part["nodes"]):
                    key = (node["month"], node["topic_id"])
                    is_start = i == 0 and parent_end is None
                    is_end = i == len(part["nodes"]) - 1 and not part["branches"]
                    is_split = (
                        part["is_split"]
                        and i == len(part["nodes"]) - 1
                        and part["branches"]
                    )
                    is_branch_start = is_branch and i == 0

                    node_type = (
                        "start"
                        if is_start
                        else (
                            "end"
                            if is_end
                            else (
                                "split"
                                if is_split
                                else "branch_start" if is_branch_start else "continue"
                            )
                        )
                    )

                    # Check if ephemeral based on doc count
                    doc_count = 0
                    month = node["month"]
                    topic_id = node["topic_id"]
                    if month in monthly_topic_info:
                        topic_row = monthly_topic_info[month][
                            monthly_topic_info[month]["Topic"] == topic_id
                        ]
                        if not topic_row.empty:
                            doc_count = topic_row.iloc[0]["Count"]

                    y_coord = y_pos * ROW_SPACING

                    layout[key] = {
                        "y": y_coord,
                        "chain_id": chain["chain_id"],
                        "lineage_id": lineage_id or chain["chain_id"],
                        "type": node_type,
                        "words": node.get("words", []),
                        "is_branch": is_branch,
                        "branch_index": branch_index if is_branch else -1,
                        "doc_count": doc_count,
                        "is_ephemeral": doc_count <= EPHEMERAL_DOC_COUNT,
                        "suppress_label": False,
                    }

                    if is_branch and i == 0 and parent_doc_count is not None:
                        # If the branch inherits the same volume as the split parent,
                        # treat it as the same topic for labeling and hide the child
                        # label to avoid duplicate names.
                        if doc_count == parent_doc_count:
                            branch_inherits_parent_label = True
                            layout[key]["suppress_label"] = True
                            if lineage_id:
                                suppressed_lineages.add(lineage_id)
                    elif is_branch:
                        layout[key]["suppress_label"] = branch_inherits_parent_label

                    if is_split:
                        split_info.append(
                            {
                                "key": key,
                                "y": y_coord,
                                "words": node.get("words", []),
                                "n_branches": len(part["branches"]),
                            }
                        )

                if part["branches"]:
                    for idx, branch in enumerate(part["branches"]):
                        y_position += 1
                        lineage_counter[lineage_id or chain["chain_id"]] += 1
                        branch_lineage = (
                            f"{lineage_id or chain['chain_id']}.b"
                            f"{lineage_counter[lineage_id or chain['chain_id']]}"
                        )
                        lineage_parent[branch_lineage] = lineage_id or chain["chain_id"]
                        process_chain_part(
                            branch,
                            y_position,
                            part["nodes"][-1] if part["nodes"] else None,
                            is_branch=True,
                            branch_index=idx,
                            lineage_id=branch_lineage,
                            parent_doc_count=layout.get(
                                (
                                    part["nodes"][-1]["month"],
                                    part["nodes"][-1]["topic_id"],
                                ),
                                {},
                            ).get("doc_count"),
                        )
                return y_pos

            process_chain_part(chain, y_position, lineage_id=chain["chain_id"])
            y_position += 1

        return layout, y_position, split_info, suppressed_lineages, lineage_parent

    (
        layout,
        total_rows,
        split_info,
        suppressed_lineages,
        lineage_parent,
    ) = calculate_layout(chains)

    if not layout:
        print("No chains to visualize")
        return None

    # Precompute total comment counts per lineage (main chain and each branch) across
    # its lifetime so split branches display distinct totals instead of sharing the
    # parent count.
    lineage_comment_totals = defaultdict(int)
    for info in layout.values():
        if info["lineage_id"] is not None:
            lineage_comment_totals[info["lineage_id"]] += info.get("doc_count", 0)

    # If a branch lineage ends up with the exact same total comment volume as its
    # parent, treat it as inherited and suppress its labels to avoid duplicating the
    # parent name in the visualization.
    for lineage_id, parent_id in lineage_parent.items():
        if (
            lineage_comment_totals.get(lineage_id, 0)
            == lineage_comment_totals.get(parent_id, -1)
        ):
            suppressed_lineages.add(lineage_id)

    # IMPROVEMENT 2: Adjust figure size and margins for better x-axis visibility
    total_height = max(1, total_rows) * ROW_SPACING
    fig, ax = plt.subplots(figsize=(20, max(10, total_height * 0.5)))

    months = sorted(monthly_representations.keys())
    month_positions = {month: i for i, month in enumerate(months)}

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    chain_colors = {}
    for i, chain in enumerate(chains):
        chain_colors[chain["chain_id"]] = colors[i % len(colors)]

    # Plot nodes
    for (month, topic_id), info in layout.items():
        x = month_positions[month]
        y = info["y"]
        color = chain_colors.get(info["chain_id"], "gray")

        markers = {
            "start": ("o", 180),
            "end": ("v", 180),
            "split": ("s", 140),
            "branch_start": ("o", 160),
            "continue": ("s", 140),
        }
        marker, base_size = markers.get(info["type"], ("s", 140))

        node_color = color
        edge_width = 2 if info["type"] == "branch_start" else 1.5

        ax.scatter(
            x,
            y,
            s=base_size,
            c=[node_color],
            marker=marker,
            edgecolors="black",
            linewidth=edge_width,
            zorder=5,
        )

        # Add labels
        if info.get("suppress_label") or info.get("lineage_id") in suppressed_lineages:
            continue

        if (
            month in monthly_representations
            and topic_id in monthly_representations[month]
        ):
            if "human_label" in monthly_representations[month][topic_id]:
                label = monthly_representations[month][topic_id]["human_label"]
            else:
                label = ", ".join(info.get("words", [])[:3])

            # IMPROVEMENT 4: Increase truncation length from 35 to 55
            if len(label) > 55:
                label = label[:52] + "..."

            total_comments = lineage_comment_totals.get(info["lineage_id"], 0)
            label_with_count = f"{label} ({total_comments})"
            label_font_weight = (
                "bold" if total_comments > N_COMMENTS_FOR_BOLD else "normal"
            )

            if info["type"] == "start":
                ax.text(
                    x - 0.1,
                    y,
                    label_with_count,
                    fontsize=7,
                    ha="right",
                    va="center",
                    style="italic",
                    fontweight=label_font_weight,
                    alpha=0.7,
                    zorder=6,
                )
            elif info["type"] == "branch_start":
                base_offset = 0.15
                vertical_nudge = 0.05
                y_offset = (
                    base_offset + vertical_nudge
                    if info["branch_index"] % 2 == 0
                    else -(base_offset - vertical_nudge)
                )
                ax.annotate(
                    f"→ {label_with_count}",
                    xy=(x, y),
                    xytext=(x + 0.3, y + y_offset),
                    fontsize=7,
                    color="black",
                    style="italic",
                    fontweight=label_font_weight,
                    alpha=0.9,
                    zorder=6,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=chain_colors.get(info["chain_id"], "gray"),
                        alpha=0.5,
                        lw=0.5,
                    ),
                )

    # IMPROVEMENT 3: Change SPLIT label color from red to black (same as other labels)
    for split in split_info:
        month, topic_id = split["key"]
        x = month_positions[month]
        y = split["y"]
        split_layout = layout.get(split["key"], {})
        chain_id = split_layout.get("chain_id")
        lineage_id = split_layout.get("lineage_id")

        if (
            month in monthly_representations
            and topic_id in monthly_representations[month]
        ):
            if "human_label" in monthly_representations[month][topic_id]:
                label = monthly_representations[month][topic_id]["human_label"]
            else:
                label = ", ".join(split["words"][:3])

            # IMPROVEMENT 4: Also increase truncation for split labels
            if len(label) > 55:
                label = label[:52] + "..."

            total_comments = lineage_comment_totals.get(lineage_id, 0)
            label_with_count = f"{label} ({total_comments})"
            label_font_weight = (
                "bold" if total_comments > N_COMMENTS_FOR_BOLD else "normal"
            )

            # IMPROVEMENT 3: Changed color from 'red' to 'black', keeping italic style
            ax.text(
                x,
                y + 0.3,
                f"SPLIT: {label_with_count}",
                fontsize=7,
                ha="center",
                va="bottom",
                style="italic",  # Keep italic style like other labels
                color="black",  # Changed from 'red' to 'black'
                alpha=0.7,  # Reduced from 0.8 to match other labels
                fontweight=label_font_weight,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="gray",  # Changed from 'red' to 'gray'
                    alpha=0.5,
                ),
                zorder=6,
            )  # Reduced alpha for subtler appearance

    # Draw connections
    for edge in network.graph.edges(data=True):
        from_node, to_node, data = edge
        from_month, from_topic = from_node.split("_")
        to_month, to_topic = to_node.split("_")

        from_key = (from_month, int(from_topic))
        to_key = (to_month, int(to_topic))

        if from_key not in layout or to_key not in layout:
            continue

        from_x = month_positions[from_month]
        from_y = layout[from_key]["y"]
        to_x = month_positions[to_month]
        to_y = layout[to_key]["y"]

        color = chain_colors.get(layout[from_key]["chain_id"], "gray")
        similarity = data.get("similarity", 0)

        # Adjust line style based on similarity threshold
        if similarity >= BRIDGING_THRESHOLD:
            linestyle = "-"
            linewidth = 2
        elif similarity >= EDGE_SIM_PLOT_THRESHOLD:
            linestyle = "--"
            linewidth = 1.5
        else:
            linestyle = ":"
            linewidth = 1

        if from_y != to_y:
            arrow = FancyArrowPatch(
                (from_x, from_y),
                (to_x, to_y),
                connectionstyle="arc3,rad=0.2",
                arrowstyle="-",
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.6,
                zorder=2,
            )
            ax.add_patch(arrow)
        else:
            ax.plot(
                [from_x, to_x],
                [from_y, to_y],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.8,
                zorder=3,
            )

    ax.set_xlim(-1.5, len(months))
    ax.set_ylim(-ROW_SPACING, total_height + ROW_SPACING)
    ax.set_xticks(range(len(months)))

    # IMPROVEMENT 2: Better x-axis label formatting
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)

    # IMPROVEMENT 1: Remove title completely
    # ax.set_title() - REMOVED

    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Topic Emergence/Branch Start",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Topic Continuation",
            markerfacecolor="gray",
            markersize=8,
            markeredgecolor="black",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            label="Topic End",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            label=f"Strong (≥{BRIDGING_THRESHOLD})",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"Moderate (≥{EDGE_SIM_PLOT_THRESHOLD})",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        framealpha=0.95,
        fontsize=9,
    )

    # IMPROVEMENT 2: Adjust bottom margin to prevent x-axis cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Add extra space at bottom for x-axis labels

    edge_pairs = []
    for from_node, to_node in network.graph.edges():
        from_month, from_topic = from_node.split("_")
        to_month, to_topic = to_node.split("_")
        from_key = (from_month, int(from_topic))
        to_key = (to_month, int(to_topic))
        if from_key in layout and to_key in layout:
            edge_pairs.append((from_key, to_key))

    layout_data = {
        "layout": layout,
        "total_rows": total_rows,
        "months": months,
        "edges": edge_pairs,
    }

    return fig, layout_data


def create_comment_count_overlay(layout_data):
    """Create a simplified count-only visualization without merge/split edges."""

    if not layout_data or not layout_data.get("layout"):
        print("No nodes to visualize for comment count overlay")
        return None

    layout = layout_data["layout"]
    total_rows = layout_data.get("total_rows", 0)
    months = layout_data.get("months", [])
    month_positions = {month: i for i, month in enumerate(months)}

    total_height = max(1, total_rows) * ROW_SPACING
    fig, ax = plt.subplots(figsize=(20, max(8, total_height * 0.45)))

    markers = {
        "start": ("o", 180),
        "end": ("v", 180),
        "split": ("s", 140),
        "branch_start": ("o", 160),
        "continue": ("s", 140),
    }

    for (month, _topic_id), info in layout.items():
        x = month_positions.get(month)
        y = info.get("y", 0)
        marker, base_size = markers.get(info.get("type", "continue"), ("s", 140))

        ax.scatter(
            x,
            y,
            s=base_size * COUNT_NODE_SIZE_SCALE,
            c="white",
            marker=marker,
            edgecolors="black",
            linewidth=1.5,
            zorder=4,
        )

        ax.text(
            x,
            y,
            str(info.get("doc_count", 0)),
            ha="center",
            va="center",
            fontsize=COUNT_NODE_FONT_SIZE,
            color="black",
            fontweight="bold",
            zorder=6,
        )

    for from_key, to_key in layout_data.get("edges", []):
        from_month, _ = from_key
        to_month, _ = to_key
        x1, y1 = month_positions[from_month], layout[from_key]["y"]
        x2, y2 = month_positions[to_month], layout[to_key]["y"]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.2, zorder=3)

    ax.set_xlim(-1.5, len(months))
    ax.set_ylim(-ROW_SPACING, total_height + ROW_SPACING)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    return fig


# Create final visualization
print("\n" + "=" * 60)
print("CREATING FINAL VISUALIZATION")
print("=" * 60)

fig, layout_snapshot = create_clean_evolution_visualization_with_labels(
    network, filtered_chains, enhanced_monthly_representations
)
plt.show()

comment_fig = create_comment_count_overlay(layout_snapshot)
if comment_fig is not None:
    plt.show()
