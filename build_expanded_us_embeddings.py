
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# AI PRICE HUNTER
# Full US ESCI Catalog - MiniLM Embeddings
# Safe, checkpointed, resumable pipeline
# ============================================================

ROOT = Path(r"D:\DLPROJECT")

INPUT_FILE = (
    ROOT / "data" / "raw" / "esci"
    / "shopping_queries_dataset_products.parquet"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "embeddings_us_full_v1"
)
CHUNKS_DIR = OUTPUT_DIR / "chunks"
CONFIG_FILE = OUTPUT_DIR / "run_config.json"

EMBEDDINGS_FILE = OUTPUT_DIR / "product_text_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "product_text_metadata.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256

PARQUET_BATCH_SIZE = 20000
PRODUCTS_PER_CHUNK = 5000
ENCODE_BATCH_SIZE = 16

FIELD_LIMITS = {
    "product_title": 500,
    "product_brand": 150,
    "product_color": 150,
    "product_bullet_point": 1800,
    "product_description": 2500,
}

FORMAT_VERSION = 1


def safe_text(value, limit):
    if value is None:
        return ""
    return " ".join(str(value).split())[:limit]


def make_product_text(row):
    fields = [
        ("Title", "product_title"),
        ("Brand", "product_brand"),
        ("Color", "product_color"),
        ("Features", "product_bullet_point"),
        ("Description", "product_description"),
    ]

    parts = []
    for label, column in fields:
        value = safe_text(row.get(column), FIELD_LIMITS[column])
        if value:
            parts.append(f"{label}: {value}")

    return " | ".join(parts) or "Title: Unknown product"


def expected_config():
    return {
        "format_version": FORMAT_VERSION,
        "input_file": str(INPUT_FILE.resolve()),
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
        "products_per_chunk": PRODUCTS_PER_CHUNK,
        "field_limits": FIELD_LIMITS,
        "locale": "us",
        "deduplication": "first occurrence per US product_id",
        "normalization": "L2",
    }


def initialize_output():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Official product catalog not found:\n{INPUT_FILE}"
        )

    if EMBEDDINGS_FILE.exists() or METADATA_FILE.exists():
        raise FileExistsError(
            "Final output already exists. Refusing to overwrite it."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    config = expected_config()

    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            saved_config = json.load(f)

        if saved_config != config:
            raise RuntimeError(
                "Configuration mismatch. Do not reuse incompatible "
                "checkpoints. Use a new output directory."
            )
    else:
        if any(CHUNKS_DIR.glob("chunk_*.npz")):
            raise RuntimeError(
                "Checkpoints exist without a run configuration. "
                "Refusing to reuse them automatically."
            )

        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def encode_with_retry(model, texts):
    batch_size = ENCODE_BATCH_SIZE

    while True:
        try:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            vectors = np.asarray(vectors, dtype=np.float32)

            if vectors.shape != (len(texts), EMBEDDING_DIM):
                raise RuntimeError(
                    f"Unexpected embedding shape: {vectors.shape}"
                )

            if not np.isfinite(vectors).all():
                raise RuntimeError("Non-finite embedding values found.")

            return vectors

        except RuntimeError as exc:
            oom = (
                "out of memory" in str(exc).lower()
                or "cuda error: out of memory" in str(exc).lower()
            )

            if not oom or batch_size <= 1:
                raise

            batch_size = max(1, batch_size // 2)
            print(f"GPU memory issue; retrying with batch size {batch_size}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def process_chunk(model, chunk_number, rows):
    checkpoint = CHUNKS_DIR / f"chunk_{chunk_number:06d}.npz"
    ids = np.asarray(
        [str(row["product_id"]) for row in rows],
        dtype=str,
    )

    # Validate and reuse a completed checkpoint.
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            old_ids = saved["product_ids"].astype(str)
            vectors = saved["embeddings"]

            if not np.array_equal(old_ids, ids):
                raise RuntimeError(
                    f"Checkpoint IDs differ: {checkpoint}"
                )

            if vectors.shape != (len(ids), EMBEDDING_DIM):
                raise RuntimeError(
                    f"Invalid checkpoint dimensions: {checkpoint}"
                )

            if not np.isfinite(vectors).all():
                raise RuntimeError(
                    f"Invalid values in checkpoint: {checkpoint}"
                )

        print(f"Chunk {chunk_number}: checkpoint validated.")
        return

    texts = [make_product_text(row) for row in rows]

    print(f"\nChunk {chunk_number}: encoding {len(rows):,} products")
    vectors = encode_with_retry(model, texts)

    # Write a temporary file, then atomically publish the checkpoint.
    temporary = CHUNKS_DIR / f"chunk_{chunk_number:06d}.tmp.npz"

    np.savez(
        temporary,
        product_ids=ids,
        embeddings=vectors,
    )
    os.replace(temporary, checkpoint)

    print(f"Saved {checkpoint.name}")


def assemble_outputs():
    checkpoints = sorted(CHUNKS_DIR.glob("chunk_*.npz"))

    if not checkpoints:
        raise RuntimeError("No checkpoints found.")

    expected_names = [
        f"chunk_{i:06d}.npz" for i in range(len(checkpoints))
    ]

    if [p.name for p in checkpoints] != expected_names:
        raise RuntimeError(
            "Checkpoint numbering has gaps. Fix before assembly."
        )

    # Validate every checkpoint and ensure globally unique IDs.
    total_rows = 0
    seen_ids = set()

    print("\nValidating all checkpoints...")

    for path in checkpoints:
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["product_ids"].astype(str)
            vectors = saved["embeddings"]

            if vectors.shape != (len(ids), EMBEDDING_DIM):
                raise RuntimeError(f"Bad checkpoint: {path}")

            if not np.isfinite(vectors).all():
                raise RuntimeError(f"Non-finite vectors: {path}")

            if len(set(ids.tolist())) != len(ids):
                raise RuntimeError(f"Duplicate IDs inside {path}")

            duplicates = seen_ids.intersection(ids.tolist())
            if duplicates:
                raise RuntimeError(
                    f"Duplicate IDs across checkpoints: "
                    f"{list(duplicates)[:5]}"
                )

            seen_ids.update(ids.tolist())
            total_rows += len(ids)

    print(f"Validated {total_rows:,} products.")

    temp_embeddings = OUTPUT_DIR / "product_text_embeddings.tmp.npy"
    temp_metadata = OUTPUT_DIR / "product_text_metadata.tmp.npz"

    for path in (temp_embeddings, temp_metadata):
        if path.exists():
            path.unlink()

    matrix = np.lib.format.open_memmap(
        temp_embeddings,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, EMBEDDING_DIM),
    )

    all_ids = np.empty(total_rows, dtype=object)
    position = 0

    try:
        for number, path in enumerate(checkpoints, start=1):
            with np.load(path, allow_pickle=False) as saved:
                ids = saved["product_ids"].astype(str)
                vectors = saved["embeddings"]

                end = position + len(ids)
                matrix[position:end] = vectors
                all_ids[position:end] = ids
                position = end

            if number % 50 == 0 or number == len(checkpoints):
                print(
                    f"Assembled {number}/{len(checkpoints)} chunks; "
                    f"{position:,} products."
                )

        matrix.flush()
        del matrix

        np.savez_compressed(
            temp_metadata,
            product_ids=np.asarray(all_ids, dtype=str),
            product_locales=np.full(total_rows, "us", dtype="<U2"),
        )

        if EMBEDDINGS_FILE.exists() or METADATA_FILE.exists():
            raise FileExistsError(
                "Final outputs appeared during assembly."
            )

        os.replace(temp_embeddings, EMBEDDINGS_FILE)
        os.replace(temp_metadata, METADATA_FILE)

    except Exception:
        try:
            del matrix
        except Exception:
            pass
        raise

    print("\n" + "=" * 60)
    print("FULL US EMBEDDING BUILD COMPLETE")
    print("=" * 60)
    print("Products:", f"{total_rows:,}")
    print("Dimension:", EMBEDDING_DIM)
    print("Embeddings:", EMBEDDINGS_FILE)
    print("Metadata:", METADATA_FILE)
    print("Checkpoints retained at:", CHUNKS_DIR)


def main():
    start_time = time.time()

    print("=" * 65)
    print("AI PRICE HUNTER - FULL US MINILM EMBEDDINGS")
    print("=" * 65)

    initialize_output()

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA available")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("WARNING: CUDA unavailable; using CPU.")

    parquet = pq.ParquetFile(INPUT_FILE)

    columns = [
        "product_id",
        "product_locale",
        "product_title",
        "product_brand",
        "product_color",
        "product_bullet_point",
        "product_description",
    ]

    missing = [c for c in columns if c not in parquet.schema_arrow.names]
    if missing:
        raise RuntimeError(f"Missing catalog columns: {missing}")

    print("Source rows:", f"{parquet.metadata.num_rows:,}")
    print("Loading:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    # The set is global so duplicates across Parquet batches are removed.
    seen_ids = set()
    pending_rows = []
    chunk_number = 0
    accepted = 0

    for batch in parquet.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=columns,
    ):
        data = batch.to_pydict()

        for i, pid_value in enumerate(data["product_id"]):
            if data["product_locale"][i] != "us" or pid_value is None:
                continue

            pid = str(pid_value)
            if pid in seen_ids:
                continue

            seen_ids.add(pid)

            row = {
                column: data[column][i]
                for column in columns
            }
            row["product_id"] = pid

            pending_rows.append(row)
            accepted += 1

            if len(pending_rows) == PRODUCTS_PER_CHUNK:
                process_chunk(model, chunk_number, pending_rows)
                chunk_number += 1
                pending_rows = []

                if chunk_number % 20 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(
                        f"Progress: {accepted:,} unique US products; "
                        f"{elapsed:.1f} minutes elapsed."
                    )

    if pending_rows:
        process_chunk(model, chunk_number, pending_rows)
        chunk_number += 1

    print("\nUnique US products:", f"{accepted:,}")
    print("Chunks:", chunk_number)

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assemble_outputs()

    print(f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
