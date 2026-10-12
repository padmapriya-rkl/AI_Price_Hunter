import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# ============================================================
# PATHS
# ============================================================

BASE = r"D:\DLPROJECT"

CATALOG = os.path.join(
    BASE,
    "downloads",
    "esci_product_catalog.parquet"
)

IMAGE_IDS = os.path.join(
    BASE,
    "data",
    "processed",
    "multimodal",
    "clip_image_product_ids.npy"
)

OUT_DIR = os.path.join(
    BASE,
    "data",
    "processed",
    "multimodal",
    "m5"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

ATTRIBUTE_MATRIX = os.path.join(
    OUT_DIR,
    "attribute_tfidf.npz"
)

ATTRIBUTE_IDS = os.path.join(
    OUT_DIR,
    "attribute_product_ids.npy"
)

VECTORIZER = os.path.join(
    OUT_DIR,
    "attribute_vectorizer.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 10000


# ============================================================
# LOAD PRODUCTS USED BY IMAGE INDEX
# ============================================================

print("=" * 60)
print("M5 ATTRIBUTE INDEX")
print("=" * 60)

print("\nLoading image product IDs...")

image_ids = np.load(
    IMAGE_IDS,
    allow_pickle=True
)

image_product_ids = set(
    str(x)
    for x in image_ids
)

print(
    f"Image-index products: "
    f"{len(image_product_ids)}"
)


# ============================================================
# STREAM ESCI CATALOG
# ============================================================

print("\nReading ESCI catalog...")

parquet_file = pq.ParquetFile(
    CATALOG
)

product_records = {}

processed = 0

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=[
        "product_id",
        "product_locale",
        "product_brand",
        "product_color"
    ]
):

    df = batch.to_pandas()

    # Only US products because the SQID/Image index
    # uses the US marketplace.
    df = df[
        df["product_locale"].astype(str).str.lower() == "us"
    ]

    for _, row in df.iterrows():

        product_id = str(
            row["product_id"]
        )

        # Keep only products present in our
        # multimodal retrieval index.
        if product_id not in image_product_ids:
            continue

        brand = row["product_brand"]

        color = row["product_color"]

        if pd.isna(brand):
            brand = ""

        if pd.isna(color):
            color = ""

        brand = str(brand).strip()

        color = str(color).strip()

        attribute_text = (
            f"brand {brand} "
            f"color {color}"
        ).strip()

        # Keep first valid catalog record.
        if product_id not in product_records:

            product_records[product_id] = (
                attribute_text
            )

    processed += len(batch)

    if processed % 100000 < BATCH_SIZE:

        print(
            f"Catalog rows processed: "
            f"{processed:,} | "
            f"Matched products: "
            f"{len(product_records):,}"
        )


print(
    f"\nMatched products: "
    f"{len(product_records)}"
)


# ============================================================
# ALIGN WITH IMAGE INDEX
# ============================================================

aligned_ids = []
attribute_texts = []

for product_id in image_ids:

    product_id = str(product_id)

    if product_id not in product_records:
        continue

    aligned_ids.append(
        product_id
    )

    attribute_texts.append(
        product_records[product_id]
    )


print(
    f"Products with attributes: "
    f"{len(aligned_ids)}"
)


# ============================================================
# TF-IDF ENCODER
# ============================================================

print("\nBuilding TF-IDF attribute representation...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

attribute_matrix = vectorizer.fit_transform(
    attribute_texts
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving M5 attribute representation...")

save_npz(
    ATTRIBUTE_MATRIX,
    attribute_matrix
)

np.save(
    ATTRIBUTE_IDS,
    np.array(
        aligned_ids,
        dtype=object
    )
)

import joblib

joblib.dump(
    vectorizer,
    VECTORIZER
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("M5 ATTRIBUTE INDEX COMPLETE")
print("=" * 60)

print(
    f"Products: "
    f"{len(aligned_ids)}"
)

print(
    f"TF-IDF dimensions: "
    f"{attribute_matrix.shape[1]}"
)

print(
    f"Non-zero values: "
    f"{attribute_matrix.nnz:,}"
)

print(
    f"Matrix saved: "
    f"{ATTRIBUTE_MATRIX}"
)

print(
    f"IDs saved: "
    f"{ATTRIBUTE_IDS}"
)

print(
    f"Vectorizer saved: "
    f"{VECTORIZER}"
)