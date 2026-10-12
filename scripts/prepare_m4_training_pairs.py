import os
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

TEST_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\sqid_esci_test.parquet"
)

M4_DIR = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\m4"
)

TRAIN_IDS_FILE = os.path.join(
    M4_DIR,
    "train_query_ids.npy"
)

OUTPUT_FILE = os.path.join(
    M4_DIR,
    "training_pairs.npz"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 5000
EMBEDDING_DIM = 768


# ============================================================
# LOAD TRAIN QUERY IDS
# ============================================================

print("=" * 70)
print("PREPARING M4 TRAINING PAIRS")
print("=" * 70)

train_query_ids = set(
    np.load(
        TRAIN_IDS_FILE,
        allow_pickle=True
    ).astype(str)
)

print(
    "Training queries:",
    len(train_query_ids)
)


# ============================================================
# STORAGE
# ============================================================

query_vectors = {}
positive_text = {}
positive_image = {}
negative_text = {}
negative_image = {}


# ============================================================
# READ TEST PARQUET IN STREAMING MODE
# ============================================================

print("\nReading SQID/ESCI data...")

parquet_file = pq.ParquetFile(
    TEST_FILE
)

columns = [
    "query_id",
    "sqid_query_clip_text_features",
    "sqid_product_clip_text_features",
    "sqid_product_clip_image_features",
    "esci_label"
]


for batch in tqdm(
    parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=columns
    ),
    desc="Preparing pairs"
):

    query_ids = batch.column(
        "query_id"
    ).to_pylist()

    query_features = batch.column(
        "sqid_query_clip_text_features"
    ).to_pylist()

    product_text_features = batch.column(
        "sqid_product_clip_text_features"
    ).to_pylist()

    product_image_features = batch.column(
        "sqid_product_clip_image_features"
    ).to_pylist()

    labels = batch.column(
        "esci_label"
    ).to_pylist()


    for (
        query_id,
        query_feature,
        product_text,
        product_image,
        label
    ) in zip(
        query_ids,
        query_features,
        product_text_features,
        product_image_features,
        labels
    ):

        if query_id is None:
            continue

        query_id = str(query_id)

        # Only training queries
        if query_id not in train_query_ids:
            continue


        # We need all required features
        if query_feature is None:
            continue

        if product_text is None:
            continue

        if product_image is None:
            continue


        try:

            q = np.asarray(
                query_feature,
                dtype=np.float32
            )

            pt = np.asarray(
                product_text,
                dtype=np.float32
            )

            pi = np.asarray(
                product_image,
                dtype=np.float32
            )

        except Exception:
            continue


        # Check dimensions
        if q.shape != (EMBEDDING_DIM,):
            continue

        if pt.shape != (EMBEDDING_DIM,):
            continue

        if pi.shape != (EMBEDDING_DIM,):
            continue


        # Normalize
        q_norm = np.linalg.norm(q)
        pt_norm = np.linalg.norm(pt)
        pi_norm = np.linalg.norm(pi)

        if q_norm == 0:
            continue

        if pt_norm == 0:
            continue

        if pi_norm == 0:
            continue


        q = q / q_norm
        pt = pt / pt_norm
        pi = pi / pi_norm


        label = str(label)


        # ====================================================
        # SAVE QUERY EMBEDDING
        # ====================================================

        if query_id not in query_vectors:

            query_vectors[
                query_id
            ] = q


        # ====================================================
        # POSITIVE = EXACT / SUBSTITUTE
        # ====================================================

        if label in ("E", "S"):

            if query_id not in positive_text:

                positive_text[
                    query_id
                ] = pt

                positive_image[
                    query_id
                ] = pi


        # ====================================================
        # NEGATIVE = IRRELEVANT / COMPLEMENT
        # ====================================================

        elif label in ("I", "C"):

            if query_id not in negative_text:

                negative_text[
                    query_id
                ] = pt

                negative_image[
                    query_id
                ] = pi


# ============================================================
# BUILD FINAL ARRAYS
# ============================================================

print("\nBuilding training arrays...")

final_query_ids = []

final_queries = []
final_pos_text = []
final_pos_image = []
final_neg_text = []
final_neg_image = []


for query_id in train_query_ids:

    if query_id not in query_vectors:
        continue

    if query_id not in positive_text:
        continue

    if query_id not in positive_image:
        continue

    if query_id not in negative_text:
        continue

    if query_id not in negative_image:
        continue


    final_query_ids.append(
        query_id
    )

    final_queries.append(
        query_vectors[query_id]
    )

    final_pos_text.append(
        positive_text[query_id]
    )

    final_pos_image.append(
        positive_image[query_id]
    )

    final_neg_text.append(
        negative_text[query_id]
    )

    final_neg_image.append(
        negative_image[query_id]
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

final_queries = np.asarray(
    final_queries,
    dtype=np.float32
)

final_pos_text = np.asarray(
    final_pos_text,
    dtype=np.float32
)

final_pos_image = np.asarray(
    final_pos_image,
    dtype=np.float32
)

final_neg_text = np.asarray(
    final_neg_text,
    dtype=np.float32
)

final_neg_image = np.asarray(
    final_neg_image,
    dtype=np.float32
)

final_query_ids = np.asarray(
    final_query_ids,
    dtype=str
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 70)
print("M4 TRAINING PAIRS READY")
print("=" * 70)

print(
    "Training queries requested:",
    len(train_query_ids)
)

print(
    "Training pairs created:",
    len(final_query_ids)
)

print(
    "Query shape:",
    final_queries.shape
)

print(
    "Positive text:",
    final_pos_text.shape
)

print(
    "Positive image:",
    final_pos_image.shape
)

print(
    "Negative text:",
    final_neg_text.shape
)

print(
    "Negative image:",
    final_neg_image.shape
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    M4_DIR,
    exist_ok=True
)

np.savez(
    OUTPUT_FILE,
    query_ids=final_query_ids,
    queries=final_queries,
    positive_text=final_pos_text,
    positive_image=final_pos_image,
    negative_text=final_neg_text,
    negative_image=final_neg_image
)


print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 70)