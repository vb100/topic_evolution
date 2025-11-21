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
from umap import UMAP
from hdbscan import HDBSCAN

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

for month_str in sorted(monthly_data.keys()):
    print(f"\nProcessing month: {month_str}")
    month_df = monthly_data[month_str]
    texts = month_df["processed_text"].tolist()

    if len(texts) < 10:
        print(f"  ⚠️ Skipping {month_str}: Only {len(texts)} comments")
        continue

    n_docs = len(texts)

    # Configure HDBSCAN with adaptive parameters
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(60, min(150, int(n_docs * 0.08))),
        min_samples=min(35, max(10, int(n_docs * 0.02))),
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
    if n_docs < 100:
        min_df_value = 2
    elif n_docs < 500:
        min_df_value = 3
    elif n_docs < 1000:
        min_df_value = 5
    else:
        min_df_value = 10

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 1),
        min_df=min_df_value,
        max_df=0.95,
        max_features=10000,
        stop_words="english",
    )

    min_topic_size_value = max(40, min(80, int(n_docs * 0.06)))

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size_value,
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=False,
    )

    try:
        topics, probs = topic_model.fit_transform(texts)
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

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        continue

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

topic_evolution = {}
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
    for t1_idx, t1_id in enumerate(topics1.keys()):
        for t2_idx, t2_id in enumerate(topics2.keys()):
            sim = similarity_matrix[t1_idx, t2_idx]
            if sim > 0:
                connections.append(
                    {
                        "from_topic": t1_id,
                        "to_topic": t2_id,
                        "similarity": sim,
                        "strong_connection": sim >= 0.5,
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
        self.similarity_threshold = similarity_threshold
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
    similarity_threshold=0.5,
    min_branch_length=2,
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
EDGE_SIM_PLOT_THRESHOLD = 0.42  # only plot edges with sim >= this
# Graph linking thresholds
TOPIC_EMB_SIM_THRESHOLD = 0.35
DOC_TO_PREV_TOPIC_THRESHOLD = 0.40
BRIDGING_THRESHOLD = 0.45
TOP_TERM_COUNT = 10
EPHEMERAL_DOC_COUNT = 6


def create_clean_evolution_visualization_with_labels(
    network, chains, monthly_representations
):
    def calculate_layout(chains):
        layout = {}
        y_position = 0
        split_info = []

        for chain in chains:
            if network.calculate_chain_longevity(chain) < 2:
                continue

            def process_chain_part(
                part, y_pos, parent_end=None, is_branch=False, branch_index=0
            ):
                nonlocal y_position

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

                    layout[key] = {
                        "y": y_pos,
                        "chain_id": chain["chain_id"],
                        "type": node_type,
                        "words": node.get("words", []),
                        "is_branch": is_branch,
                        "branch_index": branch_index if is_branch else -1,
                        "doc_count": doc_count,
                        "is_ephemeral": doc_count <= EPHEMERAL_DOC_COUNT,
                    }

                    if is_split:
                        split_info.append(
                            {
                                "key": key,
                                "y": y_pos,
                                "words": node.get("words", []),
                                "n_branches": len(part["branches"]),
                            }
                        )

                if part["branches"]:
                    for idx, branch in enumerate(part["branches"]):
                        y_position += 1
                        process_chain_part(
                            branch,
                            y_position,
                            part["nodes"][-1] if part["nodes"] else None,
                            is_branch=True,
                            branch_index=idx,
                        )
                return y_pos

            process_chain_part(chain, y_position)
            y_position += 1

        return layout, y_position, split_info

    layout, total_rows, split_info = calculate_layout(chains)

    if not layout:
        print("No chains to visualize")
        return None

    # IMPROVEMENT 2: Adjust figure size and margins for better x-axis visibility
    fig, ax = plt.subplots(figsize=(20, max(10, total_rows * 0.5)))

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

            if info["type"] == "start":
                ax.text(
                    x - 0.1,
                    y,
                    label,
                    fontsize=7,
                    ha="right",
                    va="center",
                    style="italic",
                    alpha=0.7,
                )
            elif info["type"] == "branch_start":
                y_offset = 0.15 if info["branch_index"] % 2 == 0 else -0.15
                ax.annotate(
                    f"→ {label}",
                    xy=(x, y),
                    xytext=(x + 0.3, y + y_offset),
                    fontsize=7,
                    color="black",
                    style="italic",
                    alpha=0.9,
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

            # IMPROVEMENT 3: Changed color from 'red' to 'black', keeping italic style
            ax.text(
                x,
                y + 0.3,
                f"SPLIT: {label}",
                fontsize=7,
                ha="center",
                va="bottom",
                style="italic",  # Keep italic style like other labels
                color="black",  # Changed from 'red' to 'black'
                alpha=0.7,  # Reduced from 0.8 to match other labels
                fontweight="normal",  # Changed from 'bold' to 'normal'
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="gray",  # Changed from 'red' to 'gray'
                    alpha=0.5,
                ),
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
    ax.set_ylim(-1, total_rows + 1)
    ax.set_xticks(range(len(months)))

    # IMPROVEMENT 2: Better x-axis label formatting
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)

    # IMPROVEMENT 1: Remove title completely
    # ax.set_title() - REMOVED

    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

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

    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95, fontsize=9)

    # IMPROVEMENT 2: Adjust bottom margin to prevent x-axis cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Add extra space at bottom for x-axis labels

    return fig


# Create final visualization
print("\n" + "=" * 60)
print("CREATING FINAL VISUALIZATION")
print("=" * 60)

fig = create_clean_evolution_visualization_with_labels(
    network, filtered_chains, enhanced_monthly_representations
)
plt.show()
