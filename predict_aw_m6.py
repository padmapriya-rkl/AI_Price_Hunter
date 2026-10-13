
from pathlib import Path
import re

import joblib
import pandas as pd
import numpy as np

ROOT = Path(r"D:\DLPROJECT")
DATA_DIR = ROOT / "data" / "processed" / "amazon_walmart_m6"
MODEL_PATH = DATA_DIR / "model" / "m6_model.joblib"
CATALOG_PATH = DATA_DIR / "products.parquet"


def clean(value):
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+", " ",
        re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    ).strip()


def jaccard(a, b):
    a, b = set(clean(a).split()), set(clean(b).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def equal_nonempty(a, b):
    a, b = clean(a), clean(b)
    return float(bool(a) and bool(b) and a == b)


def compact_equal(a, b):
    a, b = clean(a).replace(" ", ""), clean(b).replace(" ", "")
    return float(bool(a) and bool(b) and a == b)


def product_text(p):
    return clean(" ".join(
        str(p.get(c, ""))
        for c in ["title", "shortdescr", "longdescr", "techdetails"]
    ))


def pair_features(a, b):
    ta, tb = clean(a.get("title", "")), clean(b.get("title", ""))
    xa, xb = product_text(a), product_text(b)

    return {
        "title_jaccard": jaccard(ta, tb),
        "description_jaccard": jaccard(xa, xb),
        "brand_equal": equal_nonempty(a.get("brand", ""), b.get("brand", "")),
        "model_equal": compact_equal(a.get("modelno", ""), b.get("modelno", "")),
        "category_equal": equal_nonempty(
            a.get("category", ""), b.get("category", "")
        ),
        "dimensions_equal": equal_nonempty(
            a.get("dimensions", ""), b.get("dimensions", "")
        ),
        "same_retailer": float(
            clean(a.get("dataset", "")) == clean(b.get("dataset", ""))
        ),
        "both_have_model": float(
            bool(clean(a.get("modelno", "")))
            and bool(clean(b.get("modelno", "")))
        ),
        "both_have_brand": float(
            bool(clean(a.get("brand", "")))
            and bool(clean(b.get("brand", "")))
        ),
    }


def load_m6():
    bundle = joblib.load(MODEL_PATH)
    products = pd.read_parquet(CATALOG_PATH)
    products["id"] = products["id"].astype(str).str.strip()
    products = products.drop_duplicates("id").set_index("id", drop=False)
    return bundle, products


def predict_pair(bundle, a, b):
    features = pd.DataFrame([pair_features(a, b)])
    text_a, text_b = product_text(a), product_text(b)

    vectorizer = bundle["vectorizer"]
    va = vectorizer.transform([text_a])
    vb = vectorizer.transform([text_b])
    features["tfidf_cosine"] = np.asarray(
        va.multiply(vb).sum(axis=1)
    ).ravel()

    # Preserve the feature order used during training.
    features = features.reindex(
        columns=bundle["feature_columns"], fill_value=0.0
    )

    if bundle["winner_name"] == "text_attribute_baseline":
        probability = bundle["winner_model"].predict_proba(features)[0, 1]
    else:
        text_score = bundle["text_model"].predict_proba(
            features[bundle["text_columns"]]
        )[0, 1]
        attr_score = bundle["attribute_model"].predict_proba(
            features[bundle["attribute_columns"]]
        )[0, 1]

        router_input = np.array([[
            text_score,
            attr_score,
            features["model_equal"].iloc[0],
            features["brand_equal"].iloc[0],
        ]])

        probability = bundle["winner_model"].predict_proba(
            router_input
        )[0, 1]

    return float(probability), float(probability >= bundle["threshold"])


def main():
    bundle, products = load_m6()

    print("AI Price Hunter — M6 Product Match Tester")
    print("Model:", bundle["winner_name"])
    print("Match threshold:", round(bundle["threshold"], 3))
    print("\nEnter two product IDs from the processed catalog.")
    print("Type q to quit.")

    while True:
        id1 = input("\nAmazon product ID: ").strip()
        if id1.lower() == "q":
            break

        id2 = input("Walmart product ID: ").strip()
        if id2.lower() == "q":
            break

        if id1 not in products.index or id2 not in products.index:
            print("One or both IDs were not found.")
            continue

        a, b = products.loc[id1], products.loc[id2]

        print("\nAmazon:", a.get("title", ""))
        print("Walmart:", b.get("title", ""))

        probability, is_match = predict_pair(bundle, a, b)

        print(f"\nM6 match probability: {probability:.2%}")
        if is_match:
            print("Prediction: LIKELY MATCH")
            print("Verify product specifications before comparing prices.")
        else:
            print("Prediction: NOT A MATCH at the selected threshold.")

        print("Amazon recorded price:", a.get("price", ""))
        print("Walmart recorded price:", b.get("price", ""))


if __name__ == "__main__":
    main()
