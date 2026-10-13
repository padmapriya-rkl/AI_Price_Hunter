import os
import time
import gc

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# DLPROJECT
# Product Text Embedding using MiniLM + RTX 3050
# ============================================================

INPUT_FILE = (
    r"D:\DLPROJECT\data\processed\text_baseline"
    r"\esci_sqid_product_text.parquet"
)

OUTPUT_DIR = r"D:\DLPROJECT\data\processed\embeddings"

CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Conservative settings for RTX 3050 6 GB
PARQUET_CHUNK_SIZE = 5000
ENCODE_BATCH_SIZE = 16

EMBEDDING_DIM = 384


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    print("=" * 65)
    print("DLPROJECT - PRODUCT TEXT EMBEDDING")
    print("=" * 65)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA: Available")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("WARNING: CUDA not available. Using CPU.")

    # --------------------------------------------------------
    # Read Parquet metadata
    # --------------------------------------------------------

    parquet_file = pq.ParquetFile(INPUT_FILE)

    total_products = parquet_file.metadata.num_rows

    print(f"Total products: {total_products:,}")
    print(f"Parquet chunk size: {PARQUET_CHUNK_SIZE:,}")
    print(f"GPU batch size: {ENCODE_BATCH_SIZE}")

    # --------------------------------------------------------
    # Load MiniLM
    # --------------------------------------------------------

    print("\nLoading MiniLM...")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    # Prevent unnecessarily long sequences
    model.max_seq_length = 256

    print("MiniLM loaded successfully.")

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    total_chunks = (
        total_products + PARQUET_CHUNK_SIZE - 1
    ) // PARQUET_CHUNK_SIZE

    start_time = time.time()

    for chunk_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=PARQUET_CHUNK_SIZE,
            columns=[
                "product_id",
                "product_locale",
                "product_text"
            ]
        )
    ):

        chunk_file = os.path.join(
            CHUNKS_DIR,
            f"chunk_{chunk_number:04d}.npz"
        )

        # ----------------------------------------------------
        # Skip already completed chunks
        # ----------------------------------------------------

        if os.path.exists(chunk_file):

            print(
                f"\n[{chunk_number + 1}/{total_chunks}] "
                f"Already completed - skipping."
            )

            continue

        data = batch.to_pydict()

        product_ids = data["product_id"]
        locales = data["product_locale"]

        texts = [
            "" if x is None else str(x)
            for x in data["product_text"]
        ]

        print(
            f"\n[{chunk_number + 1}/{total_chunks}] "
            f"Encoding {len(texts):,} products..."
        )

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        chunk_start = time.time()

        embeddings = model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Make sure float32 is used
        embeddings = embeddings.astype(
            np.float32,
            copy=False
        )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        np.savez(
            chunk_file,
            embeddings=embeddings,
            product_ids=np.array(
                product_ids,
                dtype=str
            ),
            product_locales=np.array(
                locales,
                dtype=str
            )
        )

        chunk_time = time.time() - chunk_start

        print(
            f"Saved: {chunk_file}"
        )

        print(
            f"Chunk time: {chunk_time / 60:.2f} minutes"
        )

        # ----------------------------------------------------
        # GPU memory information
        # ----------------------------------------------------

        if device == "cuda":

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**2
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**2
            )

            print(
                f"GPU memory: "
                f"{allocated:.0f} MB allocated / "
                f"{reserved:.0f} MB reserved"
            )

            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Release memory
        # ----------------------------------------------------

        del embeddings
        del texts
        del data

        gc.collect()

    # --------------------------------------------------------
    # Combine chunks
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("All embedding chunks completed.")
    print("Combining chunks...")
    print("=" * 65)

    all_chunk_files = sorted(
        [
            os.path.join(CHUNKS_DIR, f)
            for f in os.listdir(CHUNKS_DIR)
            if f.endswith(".npz")
        ]
    )

    if len(all_chunk_files) != total_chunks:

        print(
            f"ERROR: Expected {total_chunks} chunks, "
            f"but found {len(all_chunk_files)}."
        )

        print(
            "No final embedding file will be created yet."
        )

        return

    final_embeddings_file = os.path.join(
        OUTPUT_DIR,
        "product_text_embeddings.npy"
    )

    final_metadata_file = os.path.join(
        OUTPUT_DIR,
        "product_text_metadata.npz"
    )

    # --------------------------------------------------------
    # Create final memory-mapped embedding file
    # --------------------------------------------------------

    final_embeddings = np.lib.format.open_memmap(
        final_embeddings_file,
        mode="w+",
        dtype=np.float32,
        shape=(total_products, EMBEDDING_DIM)
    )

    all_product_ids = []
    all_locales = []

    position = 0

    for i, chunk_file in enumerate(all_chunk_files):

        print(
            f"Combining chunk {i + 1}/{total_chunks}..."
        )

        chunk = np.load(
            chunk_file,
            allow_pickle=False
        )

        emb = chunk["embeddings"]

        count = len(emb)

        final_embeddings[
            position:position + count
        ] = emb

        all_product_ids.extend(
            chunk["product_ids"].tolist()
        )

        all_locales.extend(
            chunk["product_locales"].tolist()
        )

        position += count

        chunk.close()

    final_embeddings.flush()

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    np.savez(
        final_metadata_file,
        product_ids=np.array(
            all_product_ids,
            dtype=str
        ),
        product_locales=np.array(
            all_locales,
            dtype=str
        )
    )

    del final_embeddings
    del all_product_ids
    del all_locales

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DLPROJECT PRODUCT EMBEDDING COMPLETE")
    print("=" * 65)

    print(f"Products: {total_products:,}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    print("\nFinal files:")

    print(final_embeddings_file)
    print(final_metadata_file)

    print(
        f"\nTotal runtime: {total_time / 60:.2f} minutes"
    )

    print("\nNext stage: FAISS retrieval.")


if __name__ == "__main__":
    main()