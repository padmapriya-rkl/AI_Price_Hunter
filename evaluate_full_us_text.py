
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(r"D:\DLPROJECT")

EXAMPLES_FILE = (
    ROOT / "data" / "raw" / "esci"
    / "shopping_queries_dataset_examples.parquet"
)
INDEX_FILE = (
    ROOT / "data" / "indexes"
    / "product_text_faiss_us_full.index"
)
IDS_FILE = (
    ROOT / "data" / "indexes"
    / "product_text_faiss_us_full_ids.npy"
)
RESULTS_DIR = ROOT / "results" / "full_us_text"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
QUERY_BATCH_SIZE = 256

LABEL_GRADES = {"E": 3, "S": 2, "C": 1, "I": 0}


def dcg_at_k(relevances, k):
    rel = np.asarray(relevances[:k], dtype=np.float64)
    if len(rel) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(rel) + 2))
    gains = np.power(2.0, rel) - 1.0
    return float(np.sum(gains / discounts))


def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FULL US CATALOG - MINILM TEXT RETRIEVAL EVALUATION")
    print("=" * 65)

    # 1. Load the official US test examples.
    print("\n[1/6] Loading official ESCI test examples...")
    df = pd.read_parquet(
        EXAMPLES_FILE,
        columns=[
            "query_id", "query", "product_id",
            "product_locale", "esci_label", "split"
        ],
    )

    df = df[
        (df["split"] == "test")
        & (df["product_locale"] == "us")
    ].copy()

    if df.empty:
        raise ValueError("No US test examples found.")

    df["query_id"] = df["query_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)

    # 2. Build graded relevance maps.
    print("\n[2/6] Preparing relevance labels...")
    relevance_lookup = {}

    for row in df.itertuples(index=False):
        qid = row.query_id
        pid = row.product_id
        grade = LABEL_GRADES.get(str(row.esci_label), 0)

        relevance_lookup.setdefault(qid, {})[pid] = grade

    queries_df = (
        df[["query_id", "query"]]
        .drop_duplicates(subset=["query_id"])
        .reset_index(drop=True)
    )

    print("Unique US test queries:", f"{len(queries_df):,}")

    # 3. Load full index and its aligned product IDs.
    print("\n[3/6] Loading full FAISS index and IDs...")
    index = faiss.read_index(str(INDEX_FILE))
    product_ids = np.load(IDS_FILE, allow_pickle=False).astype(str)

    if index.ntotal != len(product_ids):
        raise ValueError("FAISS index and product ID counts differ.")

    if len(set(product_ids.tolist())) != len(product_ids):
        raise ValueError("Duplicate product IDs in index mapping.")

    if index.d != 384:
        raise ValueError(f"Unexpected index dimension: {index.d}")

    print("Products indexed:", f"{index.ntotal:,}")

    # Convert FAISS positions to product IDs.
    # The order must match the order used when building the index.
    print("Preparing product ID mapping...")
    faiss_ids = product_ids

    # 4. Load MiniLM and encode queries.
    print("\n[4/6] Loading MiniLM...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 256

    print("Encoding and searching queries...")
    all_rows = []
    total_queries = len(queries_df)

    for start in range(0, total_queries, QUERY_BATCH_SIZE):
        batch_df = queries_df.iloc[start:start + QUERY_BATCH_SIZE]
        batch_queries = batch_df["query"].astype(str).tolist()

        query_embeddings = model.encode(
            batch_queries,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embeddings = np.asarray(
            query_embeddings, dtype=np.float32
        )

        scores, positions = index.search(query_embeddings, TOP_K)

        for j, row in enumerate(batch_df.itertuples(index=False)):
            qid = str(row.query_id)
            relevance = relevance_lookup.get(qid, {})

            retrieved_ids = [
                str(faiss_ids[pos])
                for pos in positions[j]
                if 0 <= pos < len(faiss_ids)
            ]

            retrieved_grades = [
                relevance.get(pid, 0) for pid in retrieved_ids
            ]

            # Conventional Recall@10:
            # relevant E/S products retrieved / all E/S products
            relevant_ids = {
                pid for pid, grade in relevance.items()
                if grade >= 2
            }
            retrieved_relevant = sum(
                pid in relevant_ids for pid in retrieved_ids
            )

            recall = (
                retrieved_relevant / len(relevant_ids)
                if relevant_ids else 0.0
            )

            # HitRate@10: at least one E/S product in top 10.
            hit_rate = float(retrieved_relevant > 0)

            # Graded nDCG@10 using E=3, S=2, C=1, I=0.
            ideal_grades = sorted(
                relevance.values(), reverse=True
            )[:TOP_K]

            dcg = dcg_at_k(retrieved_grades, TOP_K)
            idcg = dcg_at_k(ideal_grades, TOP_K)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            all_rows.append({
                "query_id": qid,
                "query": str(row.query),
                "recall_at_10": recall,
                "hit_rate_at_10": hit_rate,
                "ndcg_at_10": ndcg,
                "relevant_es_products": len(relevant_ids),
                "retrieved_relevant_es": retrieved_relevant,
            })

        completed = min(start + len(batch_df), total_queries)
        print(f"Evaluated {completed:,}/{total_queries:,} queries")

    # 5. Save results to new files.
    print("\n[5/6] Saving results...")
    per_query = pd.DataFrame(all_rows)

    summary = pd.DataFrame([{
        "model": "MiniLM + Full US FAISS",
        "dataset": "Official ESCI US test",
        "queries_evaluated": len(per_query),
        "products_indexed": index.ntotal,
        "top_k": TOP_K,
        "recall_at_10": per_query["recall_at_10"].mean(),
        "hit_rate_at_10": per_query["hit_rate_at_10"].mean(),
        "ndcg_at_10": per_query["ndcg_at_10"].mean(),
        "runtime_minutes": (time.time() - start_time) / 60,
    }])

    summary_path = RESULTS_DIR / "full_us_text_summary.csv"
    per_query_path = RESULTS_DIR / "full_us_text_per_query.csv"

    summary.to_csv(summary_path, index=False)
    per_query.to_csv(per_query_path, index=False)

    # 6. Print final metrics.
    print("\n[6/6] Evaluation complete")
    print("=" * 65)
    print("Queries evaluated:", f"{len(per_query):,}")
    print("Products indexed:", f"{index.ntotal:,}")
    print(f"Recall@10:   {summary.iloc[0]['recall_at_10']:.4f}")
    print(f"HitRate@10:  {summary.iloc[0]['hit_rate_at_10']:.4f}")
    print(f"nDCG@10:     {summary.iloc[0]['ndcg_at_10']:.4f}")
    print(f"Runtime:     {summary.iloc[0]['runtime_minutes']:.2f} minutes")
    print("Summary:", summary_path)
    print("Per-query:", per_query_path)
    print("=" * 65)


if __name__ == "__main__":
    main()
