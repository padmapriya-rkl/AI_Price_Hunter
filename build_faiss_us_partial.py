
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(r"D:\DLPROJECT")
CHUNKS_DIR = (
    ROOT / "data" / "processed"
    / "embeddings_us_full_v1" / "chunks"
)
OUTPUT_DIR = ROOT / "data" / "indexes"
OUTPUT_INDEX = OUTPUT_DIR / "product_text_faiss_us_partial.index"
OUTPUT_IDS = OUTPUT_DIR / "product_text_faiss_us_partial_ids.npy"

DIMENSION = 384


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Never overwrite an existing index or ID mapping.
    if OUTPUT_INDEX.exists() or OUTPUT_IDS.exists():
        raise FileExistsError(
            "Partial FAISS output already exists. "
            "Choose a new filename instead of overwriting it."
        )

    chunks = sorted(CHUNKS_DIR.glob("chunk_*.npz"))

    if not chunks:
        raise FileNotFoundError(
            f"No checkpoints found in {CHUNKS_DIR}"
        )

    expected = [f"chunk_{i:06d}.npz" for i in range(len(chunks))]
    if [p.name for p in chunks] != expected:
        raise RuntimeError("Checkpoint sequence has gaps.")

    index = faiss.IndexFlatIP(DIMENSION)
    all_ids = []
    seen_ids = set()

    print(f"Found {len(chunks)} checkpoints.")

    for number, path in enumerate(chunks, start=1):
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["product_ids"].astype(str)
            vectors = np.asarray(
                saved["embeddings"], dtype=np.float32
            )

            if vectors.shape != (len(ids), DIMENSION):
                raise ValueError(f"Invalid dimensions: {path}")

            if not np.isfinite(vectors).all():
                raise ValueError(f"Non-finite vectors: {path}")

            if len(set(ids.tolist())) != len(ids):
                raise ValueError(f"Duplicate IDs inside {path}")

            overlap = seen_ids.intersection(ids.tolist())
            if overlap:
                raise ValueError(
                    f"Duplicate product IDs across chunks: "
                    f"{list(overlap)[:5]}"
                )

            # Vectors were generated with normalize_embeddings=True.
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError(
                    f"Vectors are not normalized in {path}"
                )

            index.add(np.ascontiguousarray(vectors))
            all_ids.extend(ids.tolist())
            seen_ids.update(ids.tolist())

        if number % 10 == 0 or number == len(chunks):
            print(
                f"Indexed {number}/{len(chunks)} chunks; "
                f"{index.ntotal:,} products."
            )

    ids_array = np.asarray(all_ids, dtype=str)

    if index.ntotal != len(ids_array):
        raise RuntimeError("FAISS and metadata counts do not match.")

    # Write to temporary files first.
    temp_index = OUTPUT_DIR / "product_text_faiss_us_partial.tmp.index"
    temp_ids = OUTPUT_DIR / "product_text_faiss_us_partial_ids.tmp.npy"

    faiss.write_index(index, str(temp_index))
    np.save(temp_ids, ids_array, allow_pickle=False)

    # Refuse to overwrite anything that appeared meanwhile.
    if OUTPUT_INDEX.exists() or OUTPUT_IDS.exists():
        raise FileExistsError("Final outputs appeared during build.")

    temp_index.replace(OUTPUT_INDEX)
    temp_ids.replace(OUTPUT_IDS)

    print("\nFAISS build complete.")
    print("Products indexed:", index.ntotal)
    print("Index:", OUTPUT_INDEX)
    print("Product IDs:", OUTPUT_IDS)


if __name__ == "__main__":
    main()


