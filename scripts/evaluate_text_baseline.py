import os
import time
import gc

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# DLPROJECT
# Text-only MiniLM + FAISS Baseline Evaluation
# ============================================================

VALIDATION_FILE = (
    r"D:\DLPROJECT\data\processed\splits"
    r"\validation.parquet"
)

INDEX_FILE = (
    r"D:\DLPROJECT\data\indexes"
    r"\product_text_faiss.index"
)

METADATA_FILE = (
    r"D:\DLPROJECT\data\processed\embeddings"
    r"\product_text_metadata.npz"
)

RESULTS_DIR = r"D:\DLPROJECT\results"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DCG
# ============================================================

def dcg_at_k(relevances, k):

    relevances = np.asarray(
        relevances[:k],
        dtype=np.float32
    )

    if len(relevances) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(
            2,
            len(relevances) + 2
        )
    )

    return np.sum(
        (2 ** relevances - 1)
        / discounts
    )


# ============================================================
# nDCG
# ============================================================

def ndcg_at_k(
    retrieved_relevances,
    ideal_relevances,
    k
):

    dcg = dcg_at_k(
        retrieved_relevances,
        k
    )

    ideal = sorted(
        ideal_relevances,
        reverse=True
    )

    idcg = dcg_at_k(
        ideal,
        k
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DLPROJECT - TEXT BASELINE EVALUATION")
    print("=" * 65)

    start_time = time.time()

    # --------------------------------------------------------
    # 1. Load validation data
    # --------------------------------------------------------

    print("\n[1/7] Loading validation data...")

    df = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # 2. Load FAISS index
    # --------------------------------------------------------

    print("\n[2/7] Loading FAISS index...")

    index = faiss.read_index(
        INDEX_FILE
    )

    print(
        f"Products indexed: {index.ntotal:,}"
    )

    # --------------------------------------------------------
    # 3. Load product metadata
    # --------------------------------------------------------

    print("\n[3/7] Loading product metadata...")

    metadata = np.load(
        METADATA_FILE,
        allow_pickle=False
    )

    product_ids = metadata[
        "product_ids"
    ]

    product_locales = metadata[
        "product_locales"
    ]

    print(
        f"Metadata products: {len(product_ids):,}"
    )

    # --------------------------------------------------------
    # Create FAISS-position → product key mapping
    # --------------------------------------------------------

    faiss_keys = list(
        zip(
            product_ids.astype(str),
            product_locales.astype(str)
        )
    )

    # --------------------------------------------------------
    # 4. Build relevance lookup
    # --------------------------------------------------------

    print("\n[4/7] Building relevance lookup...")

    relevance_lookup = {}

    for row in df.itertuples(
        index=False
    ):

        query_id = str(
            row.query_id
        )

        product_key = (
            str(row.product_id),
            str(row.product_locale)
        )

        # ESCI relevance:
        #
        # E = Exact      → 3
        # S = Substitute  → 2
        # C = Complement  → 1
        # I = Irrelevant  → 0
        #
        # For Recall@10 below:
        # E and S are treated as relevant.

        label_score = {
            "E": 3,
            "S": 2,
            "C": 1,
            "I": 0
        }.get(
            str(row.esci_label),
            0
        )

        if query_id not in relevance_lookup:
            relevance_lookup[query_id] = {}

        relevance_lookup[
            query_id
        ][product_key] = label_score

    # --------------------------------------------------------
    # 5. Create unique query table
    # --------------------------------------------------------

    query_table = (
        df[
            [
                "query_id",
                "query"
            ]
        ]
        .drop_duplicates(
            subset=["query_id"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Unique validation queries: "
        f"{len(query_table):,}"
    )

    # --------------------------------------------------------
    # 6. Load MiniLM
    # --------------------------------------------------------

    print("\n[5/7] Loading MiniLM...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
    )

    model.max_seq_length = 256

    print(
        "MiniLM loaded on:",
        "CUDA"
    )

    # --------------------------------------------------------
    # Prepare queries
    # --------------------------------------------------------

    queries = query_table[
        "query"
    ].tolist()

    query_ids = (
        query_table[
            "query_id"
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Encode validation queries
    # --------------------------------------------------------

    print("\n[6/7] Encoding validation queries...")

    query_embeddings = model.encode(
        queries,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embeddings = query_embeddings.astype(
        np.float32,
        copy=False
    )

    print(
        "Query embedding shape:",
        query_embeddings.shape
    )

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    print("\nSearching FAISS...")

    search_start = time.time()

    scores, indices = index.search(
        query_embeddings,
        TOP_K
    )

    search_time = time.time() - search_start

    print(
        f"FAISS search time: "
        f"{search_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    print("\nCalculating Recall@10 and nDCG@10...")

    recalls = []
    ndcgs = []

    for i, query_id in enumerate(
        query_ids
    ):

        relevance = relevance_lookup.get(
            query_id,
            {}
        )

        retrieved_relevances = []

        # --------------------------------------------
        # Get relevance of each retrieved product
        # --------------------------------------------

        for idx in indices[i]:

            if idx < 0:

                retrieved_relevances.append(
                    0
                )

                continue

            product_key = faiss_keys[
                idx
            ]

            relevance_score = relevance.get(
                product_key,
                0
            )

            retrieved_relevances.append(
                relevance_score
            )

        # --------------------------------------------
        # Recall@10
        #
        # Relevant = Exact OR Substitute
        # score >= 2
        # --------------------------------------------

        found_relevant = any(
            score >= 2
            for score in retrieved_relevances
        )

        recalls.append(
            1.0
            if found_relevant
            else 0.0
        )

        # --------------------------------------------
        # nDCG@10
        # --------------------------------------------

        ideal_relevances = list(
            relevance.values()
        )

        query_ndcg = ndcg_at_k(
            retrieved_relevances,
            ideal_relevances,
            TOP_K
        )

        ndcgs.append(
            query_ndcg
        )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    recall_at_10 = float(
        np.mean(recalls)
    )

    ndcg_at_10 = float(
        np.mean(ndcgs)
    )

    total_time = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TEXT BASELINE RESULTS")
    print("=" * 65)

    print(
        f"Model       : MiniLM + FAISS"
    )

    print(
        f"Queries     : {len(query_table):,}"
    )

    print(
        f"Products    : {index.ntotal:,}"
    )

    print(
        f"Recall@10   : {recall_at_10:.4f}"
    )

    print(
        f"nDCG@10     : {ndcg_at_10:.4f}"
    )

    print(
        f"FAISS time  : {search_time:.2f} seconds"
    )

    print(
        f"Total time  : {total_time / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "model": [
                "MiniLM + FAISS"
            ],
            "recall@10": [
                recall_at_10
            ],
            "ndcg@10": [
                ndcg_at_10
            ],
            "num_queries": [
                len(query_table)
            ],
            "num_products": [
                index.ntotal
            ],
            "top_k": [
                TOP_K
            ]
        }
    )

    output_file = os.path.join(
        RESULTS_DIR,
        "text_baseline_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Save per-query results
    # --------------------------------------------------------

    per_query_results = pd.DataFrame(
        {
            "query_id": query_ids,
            "query": queries,
            "recall@10": recalls,
            "ndcg@10": ndcgs
        }
    )

    per_query_file = os.path.join(
        RESULTS_DIR,
        "text_baseline_per_query.csv"
    )

    per_query_results.to_csv(
        per_query_file,
        index=False
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del query_embeddings
    del scores
    del indices

    gc.collect()

    if hasattr(model, "device"):
        del model

    print("\nResults saved to:")

    print(
        output_file
    )

    print(
        per_query_file
    )

    print("\n" + "=" * 65)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()