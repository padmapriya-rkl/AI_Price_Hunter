import os
import time
import numpy as np
import pandas as pd
import faiss

from tqdm import tqdm
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

IMAGE_INDEX_FILE = (
    r"D:\DLPROJECT\data\indexes\clip_image_faiss.index"
)

IMAGE_PRODUCT_IDS_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\clip_image_product_ids.npy"
)

QUERY_FEATURES_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\query_features.parquet"
)

SQID_TEST_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\sqid_esci_test.parquet"
)

RESULT_FILE = (
    r"D:\DLPROJECT\results\clip_image_baseline_results.csv"
)

PER_QUERY_RESULT_FILE = (
    r"D:\DLPROJECT\results\clip_image_baseline_per_query.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 10
BATCH_SIZE = 50000


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("CLIP IMAGE BASELINE EVALUATION")
print("=" * 70)

for file_path in [
    IMAGE_INDEX_FILE,
    IMAGE_PRODUCT_IDS_FILE,
    QUERY_FEATURES_FILE,
    SQID_TEST_FILE,
]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

print("\nAll required files found.")


# ============================================================
# LOAD CLIP IMAGE FAISS INDEX
# ============================================================

print("\nLoading CLIP image FAISS index...")

index = faiss.read_index(IMAGE_INDEX_FILE)

print("Products indexed:", index.ntotal)
print("Embedding dimension:", index.d)


# ============================================================
# LOAD IMAGE PRODUCT IDS
# ============================================================

print("\nLoading image product IDs...")

image_product_ids = np.load(
    IMAGE_PRODUCT_IDS_FILE,
    allow_pickle=True
)

image_product_ids = image_product_ids.astype(str)

print("Product IDs:", len(image_product_ids))

if len(image_product_ids) != index.ntotal:
    raise ValueError(
        "Number of product IDs does not match FAISS index size."
    )


# ============================================================
# LOAD QUERY FEATURES
# ============================================================

print("\nLoading query features...")

query_file = pq.ParquetFile(QUERY_FEATURES_FILE)

query_ids = []
query_vectors = []

for batch in query_file.iter_batches(
    batch_size=1000,
    columns=["query_id", "clip_text_features"]
):

    batch_df = batch.to_pandas()

    for _, row in batch_df.iterrows():

        qid = str(row["query_id"])
        embedding = row["clip_text_features"]

        if embedding is None:
            continue

        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32
            )
        except Exception:
            continue

        if vector.ndim != 1:
            continue

        if vector.shape[0] != index.d:
            continue

        query_ids.append(qid)
        query_vectors.append(vector)


query_ids = np.array(query_ids, dtype=str)

query_vectors = np.vstack(
    query_vectors
).astype(np.float32)


# ============================================================
# NORMALIZE QUERY EMBEDDINGS
# ============================================================

faiss.normalize_L2(query_vectors)

print("Queries:", len(query_ids))
print(
    "Query embedding shape:",
    query_vectors.shape
)


# ============================================================
# LOAD SQID / ESCI TEST GROUND TRUTH
# MEMORY-SAFE BATCH PROCESSING
# ============================================================

print("\nLoading SQID/ESCI test data...")

test_file = pq.ParquetFile(SQID_TEST_FILE)

print(
    "Test rows:",
    test_file.metadata.num_rows
)


# ------------------------------------------------------------
# Ground truth structure
#
# grades[query_id][product_id] = relevance grade
#
# E = 3
# S = 2
# C = 1
# I = 0
# ------------------------------------------------------------

label_to_grade = {
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0,
}


grades = {}


for batch in test_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=[
        "query_id",
        "product_id",
        "esci_label"
    ]
):

    batch_df = batch.to_pandas()

    batch_df["query_id"] = (
        batch_df["query_id"].astype(str)
    )

    batch_df["product_id"] = (
        batch_df["product_id"].astype(str)
    )

    for query_id, product_id, label in zip(
        batch_df["query_id"],
        batch_df["product_id"],
        batch_df["esci_label"]
    ):

        label = str(label)

        if label not in label_to_grade:
            continue

        grade = label_to_grade[label]

        if query_id not in grades:
            grades[query_id] = {}

        # In case the same product occurs more than once,
        # retain the maximum relevance grade.
        previous_grade = grades[query_id].get(
            product_id,
            0
        )

        if grade > previous_grade:
            grades[query_id][product_id] = grade


print(
    "Queries with ground truth:",
    len(grades)
)


# ============================================================
# KEEP ONLY QUERIES AVAILABLE IN BOTH DATASETS
# ============================================================

evaluation_indices = []

for i, query_id in enumerate(query_ids):

    if query_id in grades:
        evaluation_indices.append(i)


evaluation_indices = np.array(
    evaluation_indices,
    dtype=np.int64
)


query_ids_eval = query_ids[
    evaluation_indices
]

query_vectors_eval = query_vectors[
    evaluation_indices
]


print(
    "Queries available for evaluation:",
    len(query_ids_eval)
)


if len(query_ids_eval) == 0:
    raise RuntimeError(
        "No overlapping query IDs between "
        "query features and test ground truth."
    )


# ============================================================
# FAISS RETRIEVAL
# ============================================================

print("\nRunning CLIP image retrieval...")

start_time = time.time()

similarities, retrieved_indices = index.search(
    query_vectors_eval,
    TOP_K
)

faiss_time = time.time() - start_time

print(
    f"FAISS search time: {faiss_time:.2f} seconds"
)


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def dcg_at_k(relevances, k):
    """
    Compute Discounted Cumulative Gain.
    """

    relevances = relevances[:k]

    if len(relevances) == 0:
        return 0.0

    total = 0.0

    for rank, relevance in enumerate(
        relevances,
        start=1
    ):

        total += (
            (2 ** relevance - 1)
            / np.log2(rank + 1)
        )

    return total


def ndcg_at_k(retrieved_product_ids, ground_truth, k):
    """
    Compute nDCG@k using graded ESCI relevance.

    E = 3
    S = 2
    C = 1
    I = 0
    """

    retrieved_relevances = [
        ground_truth.get(
            product_id,
            0
        )
        for product_id in retrieved_product_ids[:k]
    ]

    actual_dcg = dcg_at_k(
        retrieved_relevances,
        k
    )

    ideal_relevances = sorted(
        ground_truth.values(),
        reverse=True
    )[:k]

    ideal_dcg = dcg_at_k(
        ideal_relevances,
        k
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ============================================================
# EVALUATE EACH QUERY
# ============================================================

print("\nEvaluating...")

per_query_results = []

hits = 0
ndcg_sum = 0.0


for i in tqdm(
    range(len(query_ids_eval)),
    desc="Evaluating"
):

    query_id = query_ids_eval[i]

    ground_truth = grades[query_id]

    retrieved_positions = retrieved_indices[i]

    retrieved_product_ids = []

    for position in retrieved_positions:

        if position < 0:
            continue

        product_id = image_product_ids[position]

        retrieved_product_ids.append(
            product_id
        )


    # --------------------------------------------------------
    # Recall@10 - hit style
    #
    # Hit = at least one Exact or Substitute
    # appears in Top-10.
    # --------------------------------------------------------

    relevant_products = {
        product_id
        for product_id, grade
        in ground_truth.items()
        if grade >= 2
    }


    hit = any(
        product_id in relevant_products
        for product_id in retrieved_product_ids
    )


    if hit:
        hits += 1


    # --------------------------------------------------------
    # nDCG@10
    # --------------------------------------------------------

    ndcg = ndcg_at_k(
        retrieved_product_ids,
        ground_truth,
        TOP_K
    )

    ndcg_sum += ndcg


    per_query_results.append({
        "query_id": query_id,
        "hit_at_10": int(hit),
        "ndcg_at_10": ndcg,
        "num_relevant_es": len(
            relevant_products
        ),
    })


# ============================================================
# FINAL METRICS
# ============================================================

num_queries = len(query_ids_eval)

recall_at_10 = hits / num_queries

mean_ndcg_at_10 = (
    ndcg_sum / num_queries
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("CLIP IMAGE BASELINE RESULTS")
print("=" * 70)

print(
    f"Model              : CLIP Image + FAISS"
)

print(
    f"Queries evaluated   : {num_queries:,}"
)

print(
    f"Products indexed    : {index.ntotal:,}"
)

print(
    f"Embedding dimension : {index.d}"
)

print(
    f"Recall@10           : {recall_at_10:.4f}"
)

print(
    f"nDCG@10             : {mean_ndcg_at_10:.4f}"
)

print(
    f"FAISS search time   : {faiss_time:.2f} seconds"
)

print("=" * 70)


# ============================================================
# SAVE OVERALL RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(RESULT_FILE),
    exist_ok=True
)

results_df = pd.DataFrame([
    {
        "model": "CLIP Image + FAISS",
        "queries": num_queries,
        "products": index.ntotal,
        "embedding_dim": index.d,
        "recall_at_10": recall_at_10,
        "ndcg_at_10": mean_ndcg_at_10,
        "faiss_time_seconds": faiss_time,
    }
])


results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# SAVE PER-QUERY RESULTS
# ============================================================

per_query_df = pd.DataFrame(
    per_query_results
)

per_query_df.to_csv(
    PER_QUERY_RESULT_FILE,
    index=False
)


print("\nResults saved:")
print(RESULT_FILE)
print(PER_QUERY_RESULT_FILE)

print("\nCLIP image evaluation completed successfully.")