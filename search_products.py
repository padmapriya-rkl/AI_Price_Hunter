import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# DLPROJECT - Text Retrieval
# Query -> MiniLM -> FAISS -> Top-K Products
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = (
    r"D:\DLPROJECT\data\indexes"
    r"\product_text_faiss.index"
)

METADATA_FILE = (
    r"D:\DLPROJECT\data\processed\embeddings"
    r"\product_text_metadata.npz"
)

TOP_K = 10


def main():

    print("=" * 60)
    print("DLPROJECT - PRODUCT SEARCH")
    print("=" * 60)

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print("\nLoading FAISS index...")

    index = faiss.read_index(INDEX_FILE)

    print("Products indexed:", index.ntotal)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = np.load(
        METADATA_FILE,
        allow_pickle=False
    )

    product_ids = metadata["product_ids"]
    product_locales = metadata["product_locales"]

    # --------------------------------------------------------
    # Load MiniLM
    # --------------------------------------------------------

    print("\nLoading MiniLM...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
    )

    model.max_seq_length = 256

    print("Model loaded.")

    # --------------------------------------------------------
    # Get query
    # --------------------------------------------------------

    query = input("\nEnter product query: ").strip()

    if not query:
        print("Empty query.")
        return

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding.astype(np.float32),
        TOP_K
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"TOP {TOP_K} RESULTS")
    print("=" * 60)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        print(
            f"\n{rank}. "
            f"Product ID: {product_ids[idx]}"
        )

        print(
            f"   Locale: {product_locales[idx]}"
        )

        print(
            f"   Similarity: {score:.4f}"
        )


if __name__ == "__main__":
    main()