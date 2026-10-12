
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(r"D:\DLPROJECT")
EMBEDDINGS_DIR = (
    ROOT / "data" / "processed" / "embeddings_us_full_v1"
)
INDEX_DIR = ROOT / "data" / "indexes"

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "product_text_embeddings.npy"
METADATA_FILE = EMBEDDINGS_DIR / "product_text_metadata.npz"

OUTPUT_INDEX = INDEX_DIR / "product_text_faiss_us_full.index"
OUTPUT_IDS = INDEX_DIR / "product_text_faiss_us_full_ids.npy"

EXPECTED_COUNT = 1_215_854
DIMENSION = 384


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_INDEX.exists() or OUTPUT_IDS.exists():
        raise FileExistsError(
            "Full index output already exists. "
            "Check the files before running again."
        )

    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Full embeddings or metadata file is missing."
        )

    print("Loading full embeddings...")
    vectors = np.load(EMBEDDINGS_FILE, mmap_mode="r")

    print("Loading product IDs...")
    with np.load(METADATA_FILE, allow_pickle=False) as metadata:
        ids = metadata["product_ids"].astype(str)

    print("Embedding shape:", vectors.shape)
    print("Product IDs:", len(ids))

    if vectors.shape != (EXPECTED_COUNT, DIMENSION):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")

    if len(ids) != EXPECTED_COUNT:
        raise ValueError(f"Unexpected product ID count: {len(ids)}")

    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate product IDs detected.")

    index = faiss.IndexFlatIP(DIMENSION)

    # Add in batches to limit additional RAM usage.
    batch_size = 25_000

    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch = np.asarray(vectors[start:end], dtype=np.float32)

        if not np.isfinite(batch).all():
            raise ValueError(f"Invalid embedding values in rows {start}:{end}")

        norms = np.linalg.norm(batch, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"Embeddings not normalized in rows {start}:{end}")

        index.add(np.ascontiguousarray(batch))

        print(f"Indexed {end:,}/{len(ids):,} products")

    if index.ntotal != len(ids):
        raise RuntimeError("FAISS index and product IDs do not match.")

    temp_index = INDEX_DIR / "product_text_faiss_us_full.tmp.index"
    temp_ids = INDEX_DIR / "product_text_faiss_us_full_ids.tmp.npy"

    try:
        faiss.write_index(index, str(temp_index))
        np.save(temp_ids, ids, allow_pickle=False)

        # Rename only after both temporary files have been written.
        if OUTPUT_INDEX.exists() or OUTPUT_IDS.exists():
            raise FileExistsError("Final output files appeared during build.")

        temp_index.replace(OUTPUT_INDEX)
        temp_ids.replace(OUTPUT_IDS)
    finally:
        # Clean up incomplete temporary files if the build fails.
        if temp_index.exists():
            temp_index.unlink()
        if temp_ids.exists():
            temp_ids.unlink()

    print("\nFULL US FAISS INDEX COMPLETE")
    print("Products:", index.ntotal)
    print("Index:", OUTPUT_INDEX)
    print("Product IDs:", OUTPUT_IDS)


if __name__ == "__main__":
    main()
