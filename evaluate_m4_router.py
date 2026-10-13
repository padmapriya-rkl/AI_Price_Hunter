import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE = r"D:\DLPROJECT"

DATA = os.path.join(
    BASE,
    "data",
    "processed",
    "multimodal"
)

TEST_IDS = os.path.join(
    DATA,
    "m4",
    "test_query_ids.npy"
)

PARQUET_PATH = os.path.join(
    DATA,
    "sqid_esci_test.parquet"
)

ROUTER_PATH = os.path.join(
    DATA,
    "m4",
    "adaptive_router.pt"
)

TEXT_INDEX_PATH = os.path.join(
    BASE,
    "data",
    "indexes",
    "m4_clip_text_faiss.index"
)

TEXT_IDS_PATH = os.path.join(
    DATA,
    "m4_clip_text_product_ids.npy"
)

IMAGE_INDEX_PATH = os.path.join(
    BASE,
    "data",
    "indexes",
    "clip_image_faiss.index"
)

IMAGE_IDS_PATH = os.path.join(
    DATA,
    "clip_image_product_ids.npy"
)

RESULT_PATH = os.path.join(
    BASE,
    "results",
    "m4_adaptive_text_image_results.csv"
)

PER_QUERY_PATH = os.path.join(
    BASE,
    "results",
    "m4_adaptive_text_image_per_query.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

DIM = 768
BATCH_SIZE = 5000

# Retrieve candidates from each modality.
TOP_CANDIDATES = 200

# Final ranking cutoff.
TOP_K = 10

# ESCI relevant labels.
RELEVANT_LABELS = {"E", "S"}


# ============================================================
# ADAPTIVE ROUTER
# ============================================================

class AdaptiveRouter(nn.Module):

    def __init__(self):
        super().__init__()

        # Must match the trained checkpoint.
        self.network = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        return torch.softmax(
            self.network(x),
            dim=1
        )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("M4 ADAPTIVE TEXT + IMAGE EVALUATION")
print("=" * 60)

print(f"Device: {device}")


# ============================================================
# LOAD ROUTER
# ============================================================

print("\nLoading adaptive router...")

router = AdaptiveRouter().to(device)

checkpoint = torch.load(
    ROUTER_PATH,
    map_location=device,
    weights_only=True
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

else:

    state_dict = checkpoint


router.load_state_dict(state_dict)

router.eval()

print("Router loaded successfully.")


# ============================================================
# LOAD FAISS INDEXES
# ============================================================

print("\nLoading FAISS indexes...")

text_index = faiss.read_index(
    TEXT_INDEX_PATH
)

image_index = faiss.read_index(
    IMAGE_INDEX_PATH
)

text_ids = np.load(
    TEXT_IDS_PATH,
    allow_pickle=True
)

image_ids = np.load(
    IMAGE_IDS_PATH,
    allow_pickle=True
)

print(
    f"Text products:  {text_index.ntotal}"
)

print(
    f"Image products: {image_index.ntotal}"
)


# ============================================================
# LOAD TEST QUERY IDS
# ============================================================

test_query_ids = set(
    np.load(
        TEST_IDS,
        allow_pickle=True
    ).tolist()
)

print(
    f"M4 test queries: {len(test_query_ids)}"
)


# ============================================================
# PARQUET FILE
# ============================================================

parquet_file = pq.ParquetFile(
    PARQUET_PATH
)


# ============================================================
# LOAD TEST QUERIES
# ============================================================

print("\nReading test queries...")

queries = {}

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE
):

    df = batch.to_pandas()

    for _, row in df.iterrows():

        qid = row["query_id"]

        if qid not in test_query_ids:
            continue

        if qid in queries:
            continue

        qvec = row[
            "sqid_query_clip_text_features"
        ]

        if qvec is None:
            continue

        qvec = np.array(
            qvec,
            dtype=np.float32,
            copy=True
        )

        if qvec.shape != (DIM,):
            continue

        if not np.isfinite(qvec).all():
            continue

        norm = np.linalg.norm(qvec)

        if norm == 0:
            continue

        qvec /= norm

        queries[qid] = {
            "query": row["query"],
            "query_vec": qvec
        }


print(
    f"Loaded test queries: {len(queries)}"
)


# ============================================================
# BUILD GROUND TRUTH
# ============================================================

print("\nBuilding ground truth...")

ground_truth = {}

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE
):

    df = batch.to_pandas()

    df = df[
        df["query_id"].isin(test_query_ids)
    ]

    for _, row in df.iterrows():

        label = row["esci_label"]

        if label not in RELEVANT_LABELS:
            continue

        qid = row["query_id"]

        product_key = (
            f"{row['product_id']}|||"
            f"{row['product_locale']}"
        )

        if qid not in ground_truth:
            ground_truth[qid] = set()

        ground_truth[qid].add(
            product_key
        )


print(
    f"Queries with relevant products: "
    f"{len(ground_truth)}"
)


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting M4 evaluation...")

print(
    f"Candidates per modality: "
    f"{TOP_CANDIDATES}"
)

print(
    f"Final K: {TOP_K}"
)

recall_values = []
ndcg_values = []

per_query_results = []


with torch.no_grad():

    for count, (qid, data) in enumerate(
        queries.items(),
        start=1
    ):

        qvec = data["query_vec"]

        # ----------------------------------------------------
        # ROUTER WEIGHTS
        # ----------------------------------------------------

        q_tensor = torch.from_numpy(
            qvec
        ).unsqueeze(0).to(device)

        weights = router(
            q_tensor
        )

        alpha_text = float(
            weights[0, 0].item()
        )

        alpha_image = float(
            weights[0, 1].item()
        )

        # ----------------------------------------------------
        # FAISS QUERY
        # ----------------------------------------------------

        q_search = qvec.reshape(
            1,
            DIM
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # TEXT RETRIEVAL
        # ----------------------------------------------------

        text_scores, text_indices = (
            text_index.search(
                q_search,
                TOP_CANDIDATES
            )
        )

        # ----------------------------------------------------
        # IMAGE RETRIEVAL
        # ----------------------------------------------------

        image_scores, image_indices = (
            image_index.search(
                q_search,
                TOP_CANDIDATES
            )
        )

        # ----------------------------------------------------
        # TEXT SCORES
        #
        # Text index IDs are already:
        # product_id|||locale
        # ----------------------------------------------------

        text_score_dict = {}

        for score, idx in zip(
            text_scores[0],
            text_indices[0]
        ):

            if idx < 0:
                continue

            pid = str(
                text_ids[idx]
            )

            text_score_dict[pid] = float(
                score
            )

        # ----------------------------------------------------
        # IMAGE SCORES
        #
        # IMPORTANT:
        # Image index stores only product_id.
        # Ground truth uses product_id|||locale.
        #
        # SQID evaluation is US locale, so append |||us.
        # ----------------------------------------------------

        image_score_dict = {}

        for score, idx in zip(
            image_scores[0],
            image_indices[0]
        ):

            if idx < 0:
                continue

            pid = (
                str(image_ids[idx])
                + "|||us"
            )

            image_score_dict[pid] = float(
                score
            )

        # ----------------------------------------------------
        # UNION CANDIDATES
        # ----------------------------------------------------

        candidates = (
            set(text_score_dict.keys())
            |
            set(image_score_dict.keys())
        )

        # ----------------------------------------------------
        # QUERY-ADAPTIVE FUSION
        # ----------------------------------------------------

        ranked = []

        for pid in candidates:

            text_score = text_score_dict.get(
                pid,
                0.0
            )

            image_score = image_score_dict.get(
                pid,
                0.0
            )

            adaptive_score = (
                alpha_text * text_score
                +
                alpha_image * image_score
            )

            ranked.append(
                (
                    pid,
                    adaptive_score
                )
            )

        ranked.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_results = [
            pid
            for pid, score in ranked[:TOP_K]
        ]

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        relevant = ground_truth.get(
            qid,
            set()
        )

        if not relevant:
            continue

        # ----------------------------------------------------
        # RECALL@10
        # ----------------------------------------------------

        hits = sum(
            1
            for pid in top_results
            if pid in relevant
        )

        recall = (
            1.0
            if hits > 0
            else 0.0
        )

        # ----------------------------------------------------
        # nDCG@10
        # ----------------------------------------------------

        dcg = 0.0

        for rank, pid in enumerate(
            top_results,
            start=1
        ):

            if pid in relevant:

                dcg += (
                    1.0
                    /
                    np.log2(rank + 1)
                )

        ideal_hits = min(
            len(relevant),
            TOP_K
        )

        idcg = 0.0

        for rank in range(
            1,
            ideal_hits + 1
        ):

            idcg += (
                1.0
                /
                np.log2(rank + 1)
            )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0.0
        )

        # ----------------------------------------------------
        # SAVE QUERY RESULT
        # ----------------------------------------------------

        recall_values.append(
            recall
        )

        ndcg_values.append(
            ndcg
        )

        per_query_results.append({

            "query_id": qid,

            "query": data["query"],

            "alpha_text": alpha_text,

            "alpha_image": alpha_image,

            "recall_at_10": recall,

            "ndcg_at_10": ndcg

        })

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if count % 100 == 0:

            print(
                f"Processed "
                f"{count}/{len(queries)} queries"
            )


# ============================================================
# FINAL METRICS
# ============================================================

recall = np.mean(
    recall_values
)

ndcg = np.mean(
    ndcg_values
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")

print("=" * 60)
print("M4 ADAPTIVE TEXT + IMAGE RESULTS")
print("=" * 60)

print(
    f"Queries evaluated: "
    f"{len(recall_values)}"
)

print(
    f"Products indexed: "
    f"{text_index.ntotal}"
)

print(
    f"Recall@10: "
    f"{recall:.4f}"
)

print(
    f"nDCG@10: "
    f"{ndcg:.4f}"
)

print("=" * 60)


# ============================================================
# SAVE SUMMARY
# ============================================================

os.makedirs(
    os.path.dirname(
        RESULT_PATH
    ),
    exist_ok=True
)

summary = pd.DataFrame([{

    "method":
        "M4 Adaptive Text+Image",

    "queries_evaluated":
        len(recall_values),

    "products_indexed":
        text_index.ntotal,

    "recall_at_10":
        recall,

    "ndcg_at_10":
        ndcg

}])


summary.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# SAVE PER-QUERY RESULTS
# ============================================================

per_query_df = pd.DataFrame(
    per_query_results
)

per_query_df.to_csv(
    PER_QUERY_PATH,
    index=False
)


# ============================================================
# ROUTING STATISTICS
# ============================================================

if len(per_query_df) > 0:

    print("\nRouting statistics:")

    print(
        f"Mean alpha_text: "
        f"{per_query_df['alpha_text'].mean():.4f}"
    )

    print(
        f"Mean alpha_image: "
        f"{per_query_df['alpha_image'].mean():.4f}"
    )

    print(
        f"Text-dominant queries "
        f"(alpha_text > 0.5): "
        f"{(per_query_df['alpha_text'] > 0.5).sum()}"
    )

    print(
        f"Image-dominant queries "
        f"(alpha_image > 0.5): "
        f"{(per_query_df['alpha_image'] > 0.5).sum()}"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\nResults saved:")

print(
    RESULT_PATH
)

print(
    PER_QUERY_PATH
)

print("\nM4 evaluation completed successfully.")