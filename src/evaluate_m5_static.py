
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
import joblib
from scipy.sparse import load_npz

# ============================================================
# M5: EQUAL-WEIGHT TEXT + IMAGE + ATTRIBUTE FUSION
# ============================================================

BASE = r"D:\DLPROJECT"
DATA = os.path.join(BASE, "data", "processed", "multimodal")
INDEX_DIR = os.path.join(BASE, "data", "indexes")
M4_DIR = os.path.join(DATA, "m4")
M5_DIR = os.path.join(DATA, "m5")
RESULTS_DIR = os.path.join(BASE, "results")

PARQUET_PATH = os.path.join(DATA, "sqid_esci_test.parquet")
TEST_IDS_PATH = os.path.join(M4_DIR, "test_query_ids.npy")

# FAISS indexes
TEXT_INDEX_PATH = os.path.join(
    INDEX_DIR, "m4_clip_text_faiss.index"
)
IMAGE_INDEX_PATH = os.path.join(
    INDEX_DIR, "clip_image_faiss.index"
)

# IMPORTANT: Both product-ID mappings are directly inside
# data/processed/multimodal, NOT inside the m4 folder.
TEXT_IDS_PATH = os.path.join(
    DATA, "m4_clip_text_product_ids.npy"
)
IMAGE_IDS_PATH = os.path.join(
    DATA, "clip_image_product_ids.npy"
)

# Brand + color attribute index
ATTRIBUTE_MATRIX_PATH = os.path.join(
    M5_DIR, "attribute_tfidf.npz"
)
ATTRIBUTE_IDS_PATH = os.path.join(
    M5_DIR, "attribute_product_ids.npy"
)
VECTORIZER_PATH = os.path.join(
    M5_DIR, "attribute_vectorizer.pkl"
)

# Separate output files preserve original M5 results.
SUMMARY_PATH = os.path.join(
    RESULTS_DIR, "m5_equal_weights_results.csv"
)
PER_QUERY_PATH = os.path.join(
    RESULTS_DIR, "m5_equal_weights_per_query.csv"
)

DIM = 768
BATCH_SIZE = 5000
TOP_CANDIDATES = 200
TOP_K = 10
RELEVANT_LABELS = {"E", "S"}

# Equal-weight fusion
W_TEXT = 1.0 / 3.0
W_IMAGE = 1.0 / 3.0
W_ATTRIBUTE = 1.0 / 3.0


def check_file(path):
    """Raise a clear error if a required file is missing."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def safe_vector(value):
    """Validate and L2-normalize a query embedding."""
    if value is None:
        return None

    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        return None

    norm = float(np.linalg.norm(vector))

    if norm <= 0:
        return None

    return vector / norm


def normalize_candidate_scores(score_dict):
    """
    Min-max normalize text/image scores over each modality's
    retrieved candidates.

    Attribute TF-IDF scores are intentionally left unnormalized
    in this experiment to retain the original M5 score handling.
    """
    if not score_dict:
        return {}

    values = np.asarray(
        list(score_dict.values()), dtype=np.float32
    )

    low = float(values.min())
    high = float(values.max())

    if high - low < 1e-8:
        value = float(np.clip(values[0], 0.0, 1.0))
        return {key: value for key in score_dict}

    return {
        key: float((score - low) / (high - low))
        for key, score in score_dict.items()
    }


def canonical_text_id(product_id):
    """Normalize text-index IDs to product_id|||locale format."""
    key = str(product_id)

    if "|||" in key:
        product_part, locale_part = key.split("|||", 1)
        return f"{product_part}|||{locale_part.lower()}"

    # The text index is expected to contain product IDs with
    # locale information. This fallback assumes US products.
    return f"{key}|||us"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Validate required paths
    # --------------------------------------------------------
    required_files = [
        PARQUET_PATH,
        TEST_IDS_PATH,
        TEXT_INDEX_PATH,
        TEXT_IDS_PATH,
        IMAGE_INDEX_PATH,
        IMAGE_IDS_PATH,
        ATTRIBUTE_MATRIX_PATH,
        ATTRIBUTE_IDS_PATH,
        VECTORIZER_PATH,
    ]

    print("=" * 68)
    print("M5 EQUAL-WEIGHT THREE-WAY FUSION EVALUATION")
    print("=" * 68)

    for path in required_files:
        check_file(path)

    print(f"Text weight:      {W_TEXT:.6f}")
    print(f"Image weight:     {W_IMAGE:.6f}")
    print(f"Attribute weight: {W_ATTRIBUTE:.6f}")

    # --------------------------------------------------------
    # 2. Load FAISS indexes and mappings
    # --------------------------------------------------------
    print("\nLoading FAISS indexes and product mappings...")

    text_index = faiss.read_index(TEXT_INDEX_PATH)
    image_index = faiss.read_index(IMAGE_INDEX_PATH)

    text_ids = np.load(TEXT_IDS_PATH, allow_pickle=True)
    image_ids = np.load(IMAGE_IDS_PATH, allow_pickle=True)

    if len(text_ids) != text_index.ntotal:
        raise ValueError(
            f"Text ID count ({len(text_ids)}) does not match "
            f"text index size ({text_index.ntotal})."
        )

    if len(image_ids) != image_index.ntotal:
        raise ValueError(
            f"Image ID count ({len(image_ids)}) does not match "
            f"image index size ({image_index.ntotal})."
        )

    print(f"Text index:  {text_index.ntotal:,}")
    print(f"Image index: {image_index.ntotal:,}")

    # --------------------------------------------------------
    # 3. Load attribute index
    # --------------------------------------------------------
    print("\nLoading brand/color attribute index...")

    attribute_matrix = load_npz(
        ATTRIBUTE_MATRIX_PATH
    ).tocsr()

    attribute_ids = np.load(
        ATTRIBUTE_IDS_PATH, allow_pickle=True
    )

    vectorizer = joblib.load(VECTORIZER_PATH)

    if attribute_matrix.shape[0] != len(attribute_ids):
        raise ValueError(
            "Attribute matrix row count does not match "
            "the number of attribute product IDs."
        )

    attribute_row = {
        str(product_id): row
        for row, product_id in enumerate(attribute_ids)
    }

    print(f"Attribute products: {len(attribute_ids):,}")
    print(f"Attribute matrix:   {attribute_matrix.shape}")

    # --------------------------------------------------------
    # 4. Load test-query IDs
    # --------------------------------------------------------
    print("\nLoading test query IDs...")

    test_query_ids = set(
        np.load(TEST_IDS_PATH, allow_pickle=True).tolist()
    )

    print(f"Test query IDs: {len(test_query_ids):,}")

    # --------------------------------------------------------
    # 5. Read query embeddings and ground truth
    # --------------------------------------------------------
    parquet_file = pq.ParquetFile(PARQUET_PATH)

    queries = {}
    ground_truth = {}

    required_columns = [
        "query_id",
        "query",
        "product_id",
        "product_locale",
        "esci_label",
        "sqid_query_clip_text_features",
    ]

    print("\nReading queries and relevance labels...")

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=required_columns,
    ):
        batch_df = batch.to_pandas()
        batch_df = batch_df[
            batch_df["query_id"].isin(test_query_ids)
        ]

        for row in batch_df.itertuples(index=False):
            qid = row.query_id

            if qid not in queries:
                query_vector = safe_vector(
                    row.sqid_query_clip_text_features
                )

                if query_vector is not None:
                    queries[qid] = {
                        "query": str(row.query),
                        "query_vec": query_vector,
                    }

            label = str(row.esci_label).strip()
            locale = str(row.product_locale).strip().lower()

            if label in RELEVANT_LABELS and locale == "us":
                product_key = f"{row.product_id}|||us"
                ground_truth.setdefault(qid, set()).add(
                    product_key
                )

    eval_qids = [
        qid
        for qid in queries
        if ground_truth.get(qid)
    ]

    print(f"Loaded valid queries: {len(queries):,}")
    print(
        "Queries with relevant US products: "
        f"{len(eval_qids):,}"
    )

    if not eval_qids:
        raise RuntimeError(
            "No evaluable queries. Check the parquet file, "
            "test IDs, and query embeddings."
        )

    # --------------------------------------------------------
    # 6. Retrieve candidates and fuse scores
    # --------------------------------------------------------
    per_query = []
    hit_rates = []
    recalls = []
    ndcgs = []

    print("\nRetrieving and fusing candidates...")

    for count, qid in enumerate(eval_qids, start=1):
        query_data = queries[qid]

        query_vector = query_data["query_vec"].reshape(
            1, DIM
        ).astype(np.float32)

        # Retrieve up to 200 candidates from each modality.
        text_scores, text_indices = text_index.search(
            query_vector, TOP_CANDIDATES
        )
        image_scores, image_indices = image_index.search(
            query_vector, TOP_CANDIDATES
        )

        # Text candidate scores.
        text_raw = {}

        for score, index_position in zip(
            text_scores[0], text_indices[0]
        ):
            if index_position >= 0:
                key = canonical_text_id(text_ids[index_position])
                text_raw[key] = float(score)

        # Image candidate scores.
        image_raw = {}

        for score, index_position in zip(
            image_scores[0], image_indices[0]
        ):
            if index_position >= 0:
                key = f"{image_ids[index_position]}|||us"
                image_raw[key] = float(score)

        # Normalize text and image scores independently.
        text_normalized = normalize_candidate_scores(text_raw)
        image_normalized = normalize_candidate_scores(image_raw)

        # Union of the text and image candidate sets.
        candidates = set(text_raw) | set(image_raw)

        # Brand/color attribute similarity for each candidate.
        query_attributes = vectorizer.transform(
            [query_data["query"]]
        )

        attribute_raw = {}

        for key in candidates:
            bare_product_id = key.split("|||", 1)[0]
            attribute_index = attribute_row.get(
                bare_product_id
            )

            if attribute_index is not None:
                similarity = (
                    query_attributes
                    @ attribute_matrix[attribute_index].T
                ).toarray()[0, 0]

                attribute_raw[key] = float(similarity)

        # Equal-weight fusion.
        ranked = []

        for key in candidates:
            text_score = text_normalized.get(key, 0.0)
            image_score = image_normalized.get(key, 0.0)
            attribute_score = attribute_raw.get(key, 0.0)

            fused_score = (
                W_TEXT * text_score
                + W_IMAGE * image_score
                + W_ATTRIBUTE * attribute_score
            )

            ranked.append((
                key,
                fused_score,
                text_score,
                image_score,
                attribute_score,
            ))

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        top_results = ranked[:TOP_K]
        top_product_ids = [
            result[0] for result in top_results
        ]

        relevant_products = ground_truth[qid]
        hits = sum(
            product_id in relevant_products
            for product_id in top_product_ids
        )

        # HitRate@10.
        hit_rate = float(hits > 0)

        # Recall@10.
        recall = hits / len(relevant_products)

        # Binary nDCG@10: E/S relevant, all other labels nonrelevant.
        dcg = sum(
            1.0 / np.log2(rank + 1)
            for rank, product_id in enumerate(
                top_product_ids, start=1
            )
            if product_id in relevant_products
        )

        ideal_hits = min(len(relevant_products), TOP_K)

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

        ndcg = dcg / idcg if idcg else 0.0

        hit_rates.append(hit_rate)
        recalls.append(recall)
        ndcgs.append(ndcg)

        best = (
            top_results[0]
            if top_results
            else (None, 0.0, 0.0, 0.0, 0.0)
        )

        per_query.append({
            "query_id": qid,
            "query": query_data["query"],
            "candidate_count": len(candidates),
            "relevant_product_count": len(relevant_products),
            "top10_hits": hits,
            "hit_rate_at_10": hit_rate,
            "recall_at_10": recall,
            "ndcg_at_10": ndcg,
            "top1_product_id_locale": best[0],
            "top1_fused_score": best[1],
            "top1_text_score": best[2],
            "top1_image_score": best[3],
            "top1_attribute_score": best[4],
        })

        if count % 100 == 0 or count == len(eval_qids):
            print(
                f"Processed {count:,}/{len(eval_qids):,} queries"
            )

    # --------------------------------------------------------
    # 7. Save evaluation outputs
    # --------------------------------------------------------
    results_df = pd.DataFrame(per_query)
    results_df.to_csv(PER_QUERY_PATH, index=False)

    mean_hit_rate = float(results_df["hit_rate_at_10"].mean())
    mean_recall = float(results_df["recall_at_10"].mean())
    mean_ndcg = float(results_df["ndcg_at_10"].mean())

    summary = pd.DataFrame([{
        "method": "M5 Equal-Weight Text+Image+BrandColor",
        "queries_evaluated": len(results_df),
        "text_products_indexed": text_index.ntotal,
        "image_products_indexed": image_index.ntotal,
        "attribute_products_indexed": len(attribute_ids),
        "top_candidates_per_modality": TOP_CANDIDATES,
        "top_k": TOP_K,
        "weight_text": W_TEXT,
        "weight_image": W_IMAGE,
        "weight_attribute": W_ATTRIBUTE,
        "hit_rate_at_10": mean_hit_rate,
        "recall_at_10": mean_recall,
        "ndcg_at_10": mean_ndcg,
        "relevance_definition": "E/S binary relevance",
        "split_note": (
            "Internal query partition; not official ESCI test split"
        ),
        "attribute_score_normalized": False,
    }])

    summary.to_csv(SUMMARY_PATH, index=False)

    # --------------------------------------------------------
    # 8. Print results
    # --------------------------------------------------------
    print("\n" + "=" * 68)
    print("M5 EQUAL-WEIGHT RESULTS")
    print("=" * 68)
    print(f"Queries evaluated: {len(results_df):,}")
    print(f"HitRate@10:        {mean_hit_rate:.4f}")
    print(f"Recall@10:         {mean_recall:.4f}")
    print(f"nDCG@10:           {mean_ndcg:.4f}")

    print(f"\nSummary:   {SUMMARY_PATH}")
    print(f"Per-query: {PER_QUERY_PATH}")

    print(
        "\nCaveat: attribute scores are not normalized like "
        "text/image scores."
    )
    print(
        "These results use an internal partition, not the "
        "official ESCI test split."
    )


if __name__ == "__main__":
    main()
