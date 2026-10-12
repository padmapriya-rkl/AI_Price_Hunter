import os
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss

from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

INDEX_FILE = (
    r"D:\DLPROJECT\data\indexes"
    r"\static_text_image_faiss.index"
)

PRODUCT_IDS_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\static_fusion_product_ids.npy"
)

QUERY_FEATURES_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\query_features.parquet"
)

TEST_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\sqid_esci_test.parquet"
)

RESULT_FILE = (
    r"D:\DLPROJECT\results"
    r"\static_fusion_baseline_results.csv"
)

PER_QUERY_FILE = (
    r"D:\DLPROJECT\results"
    r"\static_fusion_baseline_per_query.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 10
BATCH_SIZE = 50000


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("=" * 70)
print("M3: STATIC TEXT + IMAGE FUSION EVALUATION")
print("=" * 70)

print("\nLoading static fusion FAISS index...")

index = faiss.read_index(INDEX_FILE)

print("Products indexed:", index.ntotal)
print("Embedding dimension:", index.d)


# ============================================================
# LOAD PRODUCT IDS
# ============================================================

print("\nLoading product IDs...")

product_ids = np.load(
    PRODUCT_IDS_FILE,
    allow_pickle=True
).astype(str)

print("Product IDs:", len(product_ids))

if len(product_ids) != index.ntotal:
    raise RuntimeError(
        "Product ID count does not match FAISS index."
    )


# ============================================================
# LOAD CLIP QUERY FEATURES
# ============================================================

print("\nLoading CLIP query features...")

query_file = pq.ParquetFile(
    QUERY_FEATURES_FILE
)

query_ids = []
query_vectors = []

for batch in query_file.iter_batches(
    batch_size=1000,
    columns=[
        "query_id",
        "clip_text_features"
    ]
):

    batch_df = batch.to_pandas()

    for _, row in batch_df.iterrows():

        query_id = str(
            row["query_id"]
        )

        feature = row[
            "clip_text_features"
        ]

        if feature is None:
            continue

        try:
            vector = np.asarray(
                feature,
                dtype=np.float32
            )
        except Exception:
            continue

        if vector.ndim != 1:
            continue

        if vector.shape[0] != index.d:
            continue

        query_ids.append(query_id)
        query_vectors.append(vector)


query_ids = np.asarray(
    query_ids,
    dtype=str
)

query_vectors = np.vstack(
    query_vectors
).astype(np.float32)


# ============================================================
# NORMALIZE QUERY VECTORS
# ============================================================

faiss.normalize_L2(
    query_vectors
)

print("Queries:", len(query_ids))
print(
    "Query embedding shape:",
    query_vectors.shape
)


# ============================================================
# LOAD GROUND TRUTH
# MEMORY-SAFE
# ============================================================

print("\nLoading SQID/ESCI ground truth...")

test_file = pq.ParquetFile(
    TEST_FILE
)

print(
    "Test rows:",
    test_file.metadata.num_rows
)


# ============================================================
# ESCI RELEVANCE GRADES
# ============================================================

label_to_grade = {
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0
}

grades = {}


# ============================================================
# STREAM TEST DATA
# ============================================================

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

        old_grade = grades[
            query_id
        ].get(
            product_id,
            0
        )

        if grade > old_grade:

            grades[
                query_id
            ][product_id] = grade


print(
    "Queries with ground truth:",
    len(grades)
)


# ============================================================
# MATCH QUERY FEATURES WITH GROUND TRUTH
# ============================================================

valid_indices = [
    i
    for i, query_id in enumerate(query_ids)
    if query_id in grades
]

valid_indices = np.asarray(
    valid_indices,
    dtype=np.int64
)

query_ids_eval = query_ids[
    valid_indices
]

query_vectors_eval = query_vectors[
    valid_indices
]

print(
    "Queries available for evaluation:",
    len(query_ids_eval)
)


if len(query_ids_eval) == 0:
    raise RuntimeError(
        "No matching queries found."
    )


# ============================================================
# FAISS RETRIEVAL
# ============================================================

print("\nRunning M3 retrieval...")

start_time = time.time()

scores, indices = index.search(
    query_vectors_eval,
    TOP_K
)

faiss_time = time.time() - start_time

print(
    f"FAISS search time: {faiss_time:.2f} seconds"
)


# ============================================================
# nDCG FUNCTION
# ============================================================

def dcg_at_k(
    relevances,
    k
):

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
            /
            np.log2(rank + 1)
        )

    return total


def ndcg_at_k(
    retrieved_products,
    ground_truth,
    k
):

    retrieved_relevances = [
        ground_truth.get(
            product_id,
            0
        )
        for product_id in retrieved_products[:k]
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
# EVALUATE
# ============================================================

print("\nEvaluating...")

hits = 0
ndcg_sum = 0.0

per_query_results = []


for i in tqdm(
    range(len(query_ids_eval)),
    desc="Evaluating"
):

    query_id = query_ids_eval[i]

    ground_truth = grades[
        query_id
    ]

    retrieved_products = []

    for position in indices[i]:

        if position < 0:
            continue

        retrieved_products.append(
            product_ids[position]
        )


    # --------------------------------------------------------
    # Recall@10
    #
    # Hit if at least one Exact or Substitute
    # product appears in Top-10.
    # --------------------------------------------------------

    relevant_products = {
        product_id
        for product_id, grade
        in ground_truth.items()
        if grade >= 2
    }

    hit = any(
        product_id in relevant_products
        for product_id in retrieved_products
    )

    if hit:
        hits += 1


    # --------------------------------------------------------
    # nDCG@10
    # --------------------------------------------------------

    ndcg = ndcg_at_k(
        retrieved_products,
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
        )
    })


# ============================================================
# FINAL METRICS
# ============================================================

num_queries = len(
    query_ids_eval
)

recall_at_10 = (
    hits / num_queries
)

mean_ndcg_at_10 = (
    ndcg_sum / num_queries
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")

print("=" * 70)
print("M3 STATIC TEXT + IMAGE RESULTS")
print("=" * 70)

print(
    "Model              : Static CLIP Text + Image"
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
        "model": "Static CLIP Text + Image",
        "queries": num_queries,
        "products": index.ntotal,
        "embedding_dim": index.d,
        "recall_at_10": recall_at_10,
        "ndcg_at_10": mean_ndcg_at_10,
        "faiss_time_seconds": faiss_time
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
    PER_QUERY_FILE,
    index=False
)


print("\nResults saved:")

print(
    RESULT_FILE
)

print(
    PER_QUERY_FILE
)

print(
    "\nM3 evaluation completed successfully."
)