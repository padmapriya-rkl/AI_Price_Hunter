import os
import numpy as np
import pyarrow.parquet as pq
import faiss
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

SQID_TEST_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\sqid_esci_test.parquet"
)

INDEX_FILE = (
    r"D:\DLPROJECT\data\indexes"
    r"\static_text_image_faiss.index"
)

PRODUCT_IDS_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\static_fusion_product_ids.npy"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 5000

TEXT_WEIGHT = 0.5
IMAGE_WEIGHT = 0.5

EMBEDDING_DIM = 768


# ============================================================
# START
# ============================================================

print("=" * 70)
print("M3: STATIC TEXT + IMAGE FUSION")
print("=" * 70)

print("\nOpening SQID/ESCI parquet...")

parquet_file = pq.ParquetFile(
    SQID_TEST_FILE
)

print(
    "Total rows:",
    parquet_file.metadata.num_rows
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

index = faiss.IndexFlatIP(
    EMBEDDING_DIM
)


# ============================================================
# PRODUCT ID TRACKING
# ============================================================
#
# The test parquet contains multiple query-product rows.
# Therefore the same product can occur multiple times.
#
# We keep only the first valid occurrence of each product.
#

seen_products = set()

ordered_product_ids = []


# ============================================================
# STREAM PARQUET
# ============================================================

print("\nBuilding fused embeddings in batches...")

total_valid = 0
total_skipped = 0


batches = parquet_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=[
        "product_id",
        "sqid_product_clip_text_features",
        "sqid_product_clip_image_features",
    ]
)


for batch in tqdm(
    batches,
    desc="Processing"
):

    product_ids = batch.column(
        "product_id"
    ).to_pylist()

    text_features = batch.column(
        "sqid_product_clip_text_features"
    ).to_pylist()

    image_features = batch.column(
        "sqid_product_clip_image_features"
    ).to_pylist()


    batch_vectors = []
    batch_ids = []


    for product_id, text_feature, image_feature in zip(
        product_ids,
        text_features,
        image_features
    ):

        if product_id is None:
            total_skipped += 1
            continue

        product_id = str(product_id)


        # Skip duplicate products
        if product_id in seen_products:
            continue


        # Both modalities required
        if text_feature is None:
            total_skipped += 1
            continue

        if image_feature is None:
            total_skipped += 1
            continue


        try:

            text_vector = np.asarray(
                text_feature,
                dtype=np.float32
            )

            image_vector = np.asarray(
                image_feature,
                dtype=np.float32
            )

        except Exception:

            total_skipped += 1
            continue


        # Check dimensions

        if text_vector.ndim != 1:
            total_skipped += 1
            continue

        if image_vector.ndim != 1:
            total_skipped += 1
            continue

        if text_vector.shape[0] != EMBEDDING_DIM:
            total_skipped += 1
            continue

        if image_vector.shape[0] != EMBEDDING_DIM:
            total_skipped += 1
            continue


        # ----------------------------------------------------
        # Normalize each modality
        # ----------------------------------------------------

        text_norm = np.linalg.norm(
            text_vector
        )

        image_norm = np.linalg.norm(
            image_vector
        )


        if text_norm == 0 or image_norm == 0:
            total_skipped += 1
            continue


        text_vector = (
            text_vector / text_norm
        )

        image_vector = (
            image_vector / image_norm
        )


        # ----------------------------------------------------
        # STATIC 50/50 FUSION
        # ----------------------------------------------------

        fused_vector = (
            TEXT_WEIGHT * text_vector
            +
            IMAGE_WEIGHT * image_vector
        )


        # ----------------------------------------------------
        # Normalize fused vector
        # ----------------------------------------------------

        fused_norm = np.linalg.norm(
            fused_vector
        )


        if fused_norm == 0:
            total_skipped += 1
            continue


        fused_vector = (
            fused_vector / fused_norm
        )


        batch_vectors.append(
            fused_vector
        )

        batch_ids.append(
            product_id
        )

        seen_products.add(
            product_id
        )


    # ========================================================
    # ADD CURRENT BATCH DIRECTLY TO FAISS
    # ========================================================

    if batch_vectors:

        batch_matrix = np.asarray(
            batch_vectors,
            dtype=np.float32
        )

        faiss.normalize_L2(
            batch_matrix
        )

        index.add(
            batch_matrix
        )


        ordered_product_ids.extend(
            batch_ids
        )


        total_valid += len(
            batch_ids
        )


    # Explicitly release batch objects
    del batch_vectors
    del batch_ids
    del product_ids
    del text_features
    del image_features


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("M3 BUILD COMPLETE")
print("=" * 70)

print(
    "Products indexed:",
    index.ntotal
)

print(
    "Embedding dimension:",
    index.d
)

print(
    "Valid products:",
    total_valid
)

print(
    "Skipped rows:",
    total_skipped
)


# ============================================================
# SAVE INDEX
# ============================================================

os.makedirs(
    os.path.dirname(INDEX_FILE),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(PRODUCT_IDS_FILE),
    exist_ok=True
)


faiss.write_index(
    index,
    INDEX_FILE
)


np.save(
    PRODUCT_IDS_FILE,
    np.asarray(
        ordered_product_ids,
        dtype=str
    )
)


# ============================================================
# VERIFY
# ============================================================

saved_ids = np.load(
    PRODUCT_IDS_FILE,
    allow_pickle=True
)


print("\nSaved successfully:")

print(
    "FAISS:",
    INDEX_FILE
)

print(
    "Product IDs:",
    PRODUCT_IDS_FILE
)

print(
    "Index size:",
    index.ntotal
)

print(
    "Saved IDs:",
    len(saved_ids)
)

print("=" * 70)

print("\nM3 STATIC FUSION INDEX READY.")