import os
import numpy as np
import faiss


# ============================================================
# DLPROJECT - FAISS Product Retrieval Index
# ============================================================

EMBEDDINGS_FILE = (
    r"D:\DLPROJECT\data\processed\embeddings"
    r"\product_text_embeddings.npy"
)

METADATA_FILE = (
    r"D:\DLPROJECT\data\processed\embeddings"
    r"\product_text_metadata.npz"
)

INDEX_DIR = r"D:\DLPROJECT\data\indexes"

INDEX_FILE = os.path.join(
    INDEX_DIR,
    "product_text_faiss.index"
)


def main():

    os.makedirs(INDEX_DIR, exist_ok=True)

    print("=" * 60)
    print("DLPROJECT - BUILDING FAISS INDEX")
    print("=" * 60)

    # --------------------------------------------------------
    # Load embeddings as memory mapped array
    # --------------------------------------------------------

    print("\nLoading product embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)

    # --------------------------------------------------------
    # Create FAISS index
    #
    # Embeddings were normalized during MiniLM encoding.
    # Therefore inner product = cosine similarity.
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    print("\nCreating FAISS index...")

    index = faiss.IndexFlatIP(dimension)

    # --------------------------------------------------------
    # Add embeddings in chunks
    # --------------------------------------------------------

    chunk_size = 10000

    total = embeddings.shape[0]

    for start in range(0, total, chunk_size):

        end = min(start + chunk_size, total)

        print(
            f"Adding products {start:,} - {end:,} "
            f"of {total:,}"
        )

        batch = np.asarray(
            embeddings[start:end],
            dtype=np.float32
        )

        index.add(batch)

        del batch

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    print("\nSaving FAISS index...")

    faiss.write_index(
        index,
        INDEX_FILE
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FAISS INDEX COMPLETE")
    print("=" * 60)

    print("Products indexed:", index.ntotal)
    print("Embedding dimension:", dimension)
    print("Index:", INDEX_FILE)


if __name__ == "__main__":
    main()