import os
import numpy as np
import pyarrow.parquet as pq
import faiss
from tqdm import tqdm

PRODUCT_FILE = r"D:\DLPROJECT\data\processed\multimodal\product_features.parquet"
INDEX_FILE = r"D:\DLPROJECT\data\indexes\clip_image_faiss.index"
ID_FILE = r"D:\DLPROJECT\data\processed\multimodal\clip_image_product_ids.npy"

BATCH_SIZE = 5000
DIM = 768

print("Opening product feature file...")
pf = pq.ParquetFile(PRODUCT_FILE)

total_rows = pf.metadata.num_rows
print("Rows:", total_rows)

index = faiss.IndexFlatIP(DIM)
product_ids = []

valid_count = 0
missing_count = 0

for batch in tqdm(
    pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["product_id", "clip_image_features"]
    ),
    total=(total_rows + BATCH_SIZE - 1) // BATCH_SIZE,
    desc="Building CLIP image index"
):
    ids = batch.column(0).to_pylist()
    features = batch.column(1).to_pylist()

    valid_ids = []
    valid_features = []

    for product_id, feature in zip(ids, features):
        if feature is not None and len(feature) == DIM:
            valid_ids.append(product_id)
            valid_features.append(feature)
        else:
            missing_count += 1

    if valid_features:
        vectors = np.asarray(valid_features, dtype=np.float32)
        faiss.normalize_L2(vectors)

        index.add(vectors)
        product_ids.extend(valid_ids)

        valid_count += len(valid_features)

        del vectors

    del valid_features
    del valid_ids
    del features
    del ids

os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ID_FILE), exist_ok=True)

faiss.write_index(index, INDEX_FILE)
np.save(ID_FILE, np.asarray(product_ids, dtype=object))

print("\nDONE")
print("Total product rows:", total_rows)
print("Valid image embeddings:", valid_count)
print("Missing/invalid image embeddings:", missing_count)
print("Products indexed:", index.ntotal)
print("Embedding dimension:", index.d)
print("Index:", INDEX_FILE)
print("Product IDs:", ID_FILE)
