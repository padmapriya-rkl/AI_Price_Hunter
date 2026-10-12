
import os
import numpy as np
import pyarrow.parquet as pq
import faiss
from sentence_transformers import SentenceTransformer

BASE = r"D:\DLPROJECT"
INDEX_DIR = os.path.join(BASE, "data", "indexes")
EMBED_DIR = os.path.join(BASE, "data", "processed", "embeddings")
CATALOG = os.path.join(BASE, "downloads", "esci_product_catalog.parquet")

M1_INDEX = os.path.join(INDEX_DIR, "product_text_faiss.index")
M1_METADATA = os.path.join(EMBED_DIR, "product_text_metadata.npz")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_catalog(product_ids):
    """Read only matching product titles from the large catalog."""
    wanted = set(map(str, product_ids))
    found = {}

    parquet = pq.ParquetFile(CATALOG)

    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=["product_id", "product_locale", "product_title", "product_brand"],
    ):
        data = batch.to_pydict()

        for pid, locale, title, brand in zip(
            data["product_id"],
            data["product_locale"],
            data["product_title"],
            data["product_brand"],
        ):
            key = (str(pid), str(locale).lower())
            if key[0] in wanted and key not in found:
                found[key] = (
                    str(title or "Title unavailable"),
                    str(brand or "Brand unavailable"),
                )

        if len(found) >= len(wanted):
            break

    return found


def main():
    print("=" * 60)
    print("             AI PRICE HUNTER")
    print("         Terminal Search Prototype")
    print("=" * 60)

    print("\nLoading existing M1 index...")
    index = faiss.read_index(M1_INDEX)

    metadata = np.load(M1_METADATA, allow_pickle=False)
    product_ids = metadata["product_ids"].astype(str)
    locales = metadata["product_locales"].astype(str)

    print(f"Loaded {index.ntotal:,} indexed products.")
    print("Loading MiniLM...")
    model = SentenceTransformer(MODEL_NAME, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    model.max_seq_length = 256
    print("Ready.")

    while True:
        query = input("\nEnter a product query (or 'exit'): ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a non-empty query.")
            continue

        print("\nChoose retrieval model:")
        print("M1 - MiniLM text search")
        print("M2 - CLIP image search")
        print("M3 - Static CLIP text + image fusion")
        print("M4 - Adaptive CLIP text + image fusion")

        choice = input("Enter M1, M2, M3, or M4: ").strip().upper()

        if choice != "M1":
            print(
                f"\n{choice} cannot yet search arbitrary text queries safely: "
                "the compatible CLIP query encoder has not been identified."
            )
            
            continue

        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = index.search(query_vector, 10)
        result_indices = indices[0]
        result_ids = product_ids[result_indices]
        catalog = load_catalog(result_ids)

        print("\n" + "=" * 60)
        print("TOP 10 RESULTS — M1")
        print("Score = MiniLM/FAISS cosine-style similarity")
        print("=" * 60)

        for rank, (score, idx) in enumerate(zip(scores[0], result_indices), 1):
            pid = str(product_ids[idx])
            locale = str(locales[idx]).lower()
            title, brand = catalog.get(
                (pid, locale),
                catalog.get((pid, "us"), ("Title unavailable", "Brand unavailable")),
            )

            print(f"\n{rank}. {title}")
            print(f"   Brand: {brand}")
            print(f"   Product ID: {pid}")
            print(f"   Locale: {locale}")
            print(f"   Similarity score: {score:.4f}")


if __name__ == "__main__":
    main()
