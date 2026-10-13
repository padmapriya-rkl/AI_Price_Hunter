import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss

BASE = r"D:\DLPROJECT"

INPUT = os.path.join(
    BASE,
    "data",
    "processed",
    "multimodal",
    "sqid_esci_test.parquet"
)

INDEX_OUT = os.path.join(
    BASE,
    "data",
    "indexes",
    "m4_clip_text_faiss.index"
)

IDS_OUT = os.path.join(
    BASE,
    "data",
    "processed",
    "multimodal",
    "m4_clip_text_product_ids.npy"
)

BATCH_SIZE = 5000
DIM = 768

index = faiss.IndexFlatIP(DIM)

seen = set()
product_ids = []

parquet_file = pq.ParquetFile(INPUT)

print("Building M4 CLIP-text FAISS index...")
print(f"Input: {INPUT}")

for batch_no, batch in enumerate(
    parquet_file.iter_batches(batch_size=BATCH_SIZE)
):
    df = batch.to_pandas()

    vectors = []
    ids = []

    for _, row in df.iterrows():

        product_id = row["product_id"]
        locale = row["product_locale"]

        key = f"{product_id}|||{locale}"

        # Keep only products that have both text and image features.
        text_vec = row["sqid_product_clip_text_features"]
        image_vec = row["sqid_product_clip_image_features"]

        if text_vec is None or image_vec is None:
            continue

        if key in seen:
            continue

        text_vec = np.array(text_vec, dtype=np.float32, copy=True)
        image_vec = np.array(image_vec, dtype=np.float32, copy=True)

        if text_vec.shape != (DIM,):
            continue

        if image_vec.shape != (DIM,):
            continue

        if not np.isfinite(text_vec).all():
            continue

        if not np.isfinite(image_vec).all():
            continue

        # Normalize text vector for cosine/IP search.
        norm = np.linalg.norm(text_vec)

        if norm == 0:
            continue

        text_vec /= norm

        vectors.append(text_vec)
        ids.append(key)

        seen.add(key)

    if vectors:
        vectors = np.stack(vectors).astype(np.float32)

        index.add(vectors)
        product_ids.extend(ids)

    if batch_no % 10 == 0:
        print(
            f"Processed batches: {batch_no + 1} | "
            f"Indexed products: {index.ntotal}"
        )

print("\nSaving index...")

faiss.write_index(index, INDEX_OUT)

np.save(
    IDS_OUT,
    np.array(product_ids, dtype=object)
)

print("\nDONE")
print(f"Products indexed: {index.ntotal}")
print(f"Dimension: {index.d}")
print(f"Index saved: {INDEX_OUT}")
print(f"Product IDs saved: {IDS_OUT}")