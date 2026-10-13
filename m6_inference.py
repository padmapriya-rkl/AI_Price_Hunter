
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
import joblib
import torch
import torch.nn as nn
from scipy.sparse import load_npz


# ============================================================
# CONFIGURATION
# ============================================================

BASE = r"D:\DLPROJECT"
DATA = os.path.join(BASE, "data", "processed", "multimodal")
M5 = os.path.join(DATA, "m5")

CHECKPOINT = os.path.join(M5, "m6_adaptive_router.pt")

TEXT_INDEX = os.path.join(
    BASE, "data", "indexes", "m4_clip_text_faiss.index"
)
TEXT_IDS = os.path.join(DATA, "m4_clip_text_product_ids.npy")

IMAGE_INDEX = os.path.join(
    BASE, "data", "indexes", "clip_image_faiss.index"
)
IMAGE_IDS = os.path.join(DATA, "clip_image_product_ids.npy")

ATTR_MATRIX = os.path.join(M5, "attribute_tfidf.npz")
ATTR_IDS = os.path.join(M5, "attribute_product_ids.npy")
VECTORIZER = os.path.join(M5, "attribute_vectorizer.pkl")

CATALOG = os.path.join(
    BASE, "downloads", "esci_product_catalog.parquet"
)

QUERY_FEATURES = os.path.join(DATA, "query_features.parquet")

DIM = 768
TOP_CANDIDATES = 200
TOP_K = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# M6 ADAPTIVE ROUTER
# ============================================================

class AdaptiveRouter3(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, query):
        return torch.softmax(self.network(query), dim=1)


# ============================================================
# VECTOR UTILITIES
# ============================================================

def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)

    if vector.shape != (DIM,):
        raise ValueError(
            f"Expected embedding dimension {DIM}; got {vector.shape}"
        )

    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains invalid numeric values.")

    norm = float(np.linalg.norm(vector))

    if norm < 1e-12:
        raise ValueError("Embedding has zero norm.")

    return vector / norm


def normalize_scores(scores):
    if not scores:
        return {}

    values = np.asarray(list(scores.values()), dtype=np.float32)
    low = float(values.min())
    high = float(values.max())

    if high - low < 1e-8:
        return {key: 0.0 for key in scores}

    return {
        key: float((value - low) / (high - low))
        for key, value in scores.items()
    }


# ============================================================
# M6 INFERENCE
# ============================================================

class M6Inference:
    def __init__(self):
        print("\nLoading M6 checkpoint...")

        self.model = AdaptiveRouter3().to(DEVICE)

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
            else checkpoint
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("Checkpoint loaded.")
        if isinstance(checkpoint, dict):
            print(
                "Best epoch:",
                checkpoint.get("best_epoch", "unknown"),
            )

        print("\nLoading FAISS indexes...")

        self.text_index = faiss.read_index(TEXT_INDEX)
        self.image_index = faiss.read_index(IMAGE_INDEX)

        self.text_ids = np.load(TEXT_IDS, allow_pickle=True)
        self.image_ids = np.load(IMAGE_IDS, allow_pickle=True)

        if len(self.text_ids) != self.text_index.ntotal:
            raise ValueError(
                "Text FAISS index and product ID mapping mismatch."
            )

        if len(self.image_ids) != self.image_index.ntotal:
            raise ValueError(
                "Image FAISS index and product ID mapping mismatch."
            )

        print("Text index:", self.text_index.ntotal, "vectors")
        print("Image index:", self.image_index.ntotal, "vectors")

        print("\nLoading attribute index...")

        self.attribute_matrix = load_npz(ATTR_MATRIX).tocsr()
        self.attribute_ids = np.load(
            ATTR_IDS, allow_pickle=True
        )
        self.vectorizer = joblib.load(VECTORIZER)

        if self.attribute_matrix.shape[0] != len(self.attribute_ids):
            raise ValueError(
                "Attribute matrix and product ID mapping mismatch."
            )

        self.attribute_rows = {
            str(product_id): row
            for row, product_id in enumerate(self.attribute_ids)
        }

        print(
            "Attribute products:",
            self.attribute_matrix.shape[0],
        )

        print("\nLoading product catalog...")
        self.catalog = self.load_catalog()
        print("Catalog rows:", len(self.catalog))

        print("\nM6 is ready.")
        print("Device:", DEVICE)

    # --------------------------------------------------------
    # PRODUCT CATALOG
    # --------------------------------------------------------

    def load_catalog(self):
        if not os.path.exists(CATALOG):
            raise FileNotFoundError(
                f"Product catalog not found:\n{CATALOG}"
            )

        columns = [
            "product_id",
            "product_locale",
            "product_title",
            "product_brand",
            "product_color",
        ]

        catalog = pd.read_parquet(
            CATALOG,
            columns=columns,
        )

        catalog["product_id"] = (
            catalog["product_id"].astype(str)
        )

        catalog["product_locale"] = (
            catalog["product_locale"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        catalog = catalog.drop_duplicates(
            subset=["product_id", "product_locale"],
            keep="first",
        )

        catalog = catalog.set_index(
            ["product_id", "product_locale"]
        )

        return catalog

    def get_product_details(self, product_key):
        if "|||" in product_key:
            product_id, locale = product_key.split("|||", 1)
            locale = locale.strip().lower()
        else:
            product_id = product_key
            locale = "us"

        row = None

        if (product_id, locale) in self.catalog.index:
            row = self.catalog.loc[(product_id, locale)]

        # If exact locale is missing, try any locale for this product.
        if row is None:
            try:
                matches = self.catalog.xs(
                    product_id,
                    level="product_id",
                    drop_level=False,
                )
                if not matches.empty:
                    row = matches.iloc[0]
            except KeyError:
                pass

        if row is None:
            return {
                "product_id": product_id,
                "locale": locale,
                "title": "Product title unavailable",
                "brand": "Unknown",
                "color": "Unknown",
            }

        def clean(value):
            if pd.isna(value) or not str(value).strip():
                return "Unknown"
            return str(value).strip()

        return {
            "product_id": product_id,
            "locale": locale,
            "title": clean(row["product_title"]),
            "brand": clean(row["product_brand"]),
            "color": clean(row["product_color"]),
        }

    # --------------------------------------------------------
    # M6 SEARCH
    # --------------------------------------------------------

    def search(
        self,
        query_embedding,
        query_text="",
        top_k=TOP_K,
    ):
        query = normalize_vector(query_embedding)
        query_batch = query.reshape(1, DIM)

        # Adaptive modality weights
        with torch.no_grad():
            query_tensor = torch.from_numpy(query).unsqueeze(0)
            query_tensor = query_tensor.to(DEVICE)

            weights = (
                self.model(query_tensor)[0]
                .cpu()
                .numpy()
            )

        # Retrieve text candidates
        text_scores, text_positions = self.text_index.search(
            query_batch,
            TOP_CANDIDATES,
        )

        # Retrieve image candidates
        image_scores, image_positions = self.image_index.search(
            query_batch,
            TOP_CANDIDATES,
        )

        text_results = {}

        for score, position in zip(
            text_scores[0], text_positions[0]
        ):
            if position < 0:
                continue

            key = str(self.text_ids[position])

            if "|||" in key:
                product_id, locale = key.split("|||", 1)
                key = f"{product_id}|||{locale.lower()}"

            text_results[key] = float(score)

        image_results = {}

        for score, position in zip(
            image_scores[0], image_positions[0]
        ):
            if position < 0:
                continue

            key = f"{self.image_ids[position]}|||us"
            image_results[key] = float(score)

        # Normalize modality scores over retrieved candidates.
        text_results = normalize_scores(text_results)
        image_results = normalize_scores(image_results)

        candidates = set(text_results) | set(image_results)

        # Attribute similarity
        attribute_results = {}

        if query_text.strip():
            query_attributes = self.vectorizer.transform(
                [query_text]
            ).tocsr()

            for key in candidates:
                product_id = key.split("|||", 1)[0]
                row = self.attribute_rows.get(product_id)

                if row is not None:
                    attribute_results[key] = float(
                        (
                            query_attributes
                            @ self.attribute_matrix[row].T
                        ).toarray()[0, 0]
                    )

        # Three-way adaptive fusion
        ranked = []

        for key in candidates:
            text_score = text_results.get(key, 0.0)
            image_score = image_results.get(key, 0.0)
            attribute_score = attribute_results.get(key, 0.0)

            fused_score = float(
                weights[0] * text_score
                + weights[1] * image_score
                + weights[2] * attribute_score
            )

            details = self.get_product_details(key)

            ranked.append({
                **details,
                "product_key": key,
                "score": fused_score,
                "text_score": text_score,
                "image_score": image_score,
                "attribute_score": attribute_score,
            })

        ranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return weights, ranked[:top_k]

    # --------------------------------------------------------
    # CATALOG TITLE SEARCH
    # --------------------------------------------------------

    
    def search_catalog_by_name(self, term, limit=10):
        term = term.strip()

        if not term:
            return self.catalog.iloc[0:0]

        titles = self.catalog["product_title"].fillna("").astype(str)

        # Prioritize US products, then search other locales if necessary.
        us_mask = (
            self.catalog["product_locale"].astype(str).str.lower() == "us"
        )
        match_mask = titles.str.contains(term, case=False, regex=False)

        matches = self.catalog[us_mask & match_mask]

        if matches.empty:
            matches = self.catalog[match_mask]

        return matches.head(limit)



    # ============================================================
    # DISPLAY
    # ============================================================

def display_results(weights, results):
    print("\n" + "=" * 72)
    print("M6 PRODUCT MATCH RESULTS")
    print("=" * 72)

    print(
        "Adaptive weights | "
        f"Text: {weights[0]:.4f} | "
        f"Image: {weights[1]:.4f} | "
        f"Attributes: {weights[2]:.4f}"
    )

    if not results:
        print("\nNo candidates found.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"\n{rank}. {item['title']}")
        print(f"   Brand: {item['brand']}")
        print(f"   Color: {item['color']}")
        print(
            f"   Product ID: {item['product_id']} | "
            f"Locale: {item['locale']}"
        )
        print(f"   M6 score: {item['score']:.4f}")
        print(
            f"   Text: {item['text_score']:.4f} | "
            f"Image: {item['image_score']:.4f} | "
            f"Attributes: {item['attribute_score']:.4f}"
        )


    def display_catalog_matches(matches):
        if matches.empty:
            print("\nNo matching product names found.")
            return

        print(f"\nShowing {len(matches)} matching catalog rows:")

        for (product_id, locale), row in matches.iterrows():
            print(f"\nProduct: {row['product_title']}")
            print(f"Brand: {row['product_brand']}")
            print(f"Color: {row['product_color']}")
            print(f"Product ID: {product_id} | Locale: {locale}")


    # ============================================================
    # TERMINAL APPLICATION
    # ============================================================

def main():
    print("\n" + "=" * 72)
    print("                 AI PRICE HUNTER")
    print("             M6 Adaptive Retrieval")
    print("=" * 72)

    try:
        m6 = M6Inference()
    except Exception as error:
        print("\nCould not initialize M6.")
        print("Error:", error)
        return

    while True:
        print("\n" + "-" * 60)
        print("MAIN MENU")
        print("-" * 60)
        print("1. Run M6 using a saved query embedding")
        print("2. Search catalog by product name")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "0":
            print("\nExiting AI Price Hunter.")
            break

        elif choice == "1":
            try:
                batch = next(
                    pq.ParquetFile(QUERY_FEATURES).iter_batches(
                        batch_size=1,
                        columns=["query_id", "clip_text_features"],
                    )
                )

                query_id = batch.column(
                    "query_id"
                )[0].as_py()

                embedding = batch.column(
                    "clip_text_features"
                )[0].as_py()

                print(f"\nUsing saved query ID: {query_id}")

                weights, results = m6.search(
                    embedding,
                    query_text="",
                    top_k=TOP_K,
                )

                display_results(weights, results)

            except Exception as error:
                print("\nSearch failed:", error)

        elif choice == "2":
            term = input(
                "\nEnter product name or keyword: "
            ).strip()

            if not term:
                print("Please enter a product name.")
                continue

            try:
                matches = m6.search_catalog_by_name(
                    term,
                    limit=10,
                )
                display_catalog_matches(matches)

            except Exception as error:
                print("\nCatalog search failed:", error)

        else:
            print("Invalid choice. Enter 0, 1, or 2.")


if __name__ == "__main__":
    main()
