
from pathlib import Path
import sys
import re
import pandas as pd

ROOT = Path(r"D:\DLPROJECT")
DATA_DIR = ROOT / "data" / "processed" / "amazon_walmart_m6"

sys.path.insert(0, str(ROOT / "scripts"))

from predict_aw_m6 import load_m6, predict_pair


def normalize(value):
    return re.sub(
        r"\s+", " ",
        re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    ).strip()


def main():
    bundle, products = load_m6()

    # Keep only supported retailers.
    products = products[
        products["dataset"].astype(str).str.lower().isin(
            ["amazon", "walmart"]
        )
    ].copy()

    products["search_text"] = (
        products["title"].fillna("").astype(str)
        + " "
        + products["brand"].fillna("").astype(str)
        + " "
        + products["modelno"].fillna("").astype(str)
        + " "
        + products["shortdescr"].fillna("").astype(str)
    ).map(normalize)

    print("\nAI PRICE HUNTER — M6")
    print("Model:", bundle["winner_name"])
    print("Enter a product name or model number. Type q to quit.")

    while True:
        query = input("\nProduct to search: ").strip()
        if query.lower() == "q":
            break
        if len(query) < 2:
            print("Please enter at least two characters.")
            continue

        q = normalize(query)
        terms = set(q.split())

        # Retrieve candidates using title and attribute text.
        products["_search_score"] = products["search_text"].map(
            lambda text: (
                sum(term in set(text.split()) for term in terms)
                / max(len(terms), 1)
            )
        )

        candidates = products[
            products["_search_score"] > 0
        ].sort_values("_search_score", ascending=False).head(40)

        amazon = candidates[
            candidates["dataset"].str.lower() == "amazon"
        ].head(15)
        walmart = candidates[
            candidates["dataset"].str.lower() == "walmart"
        ].head(15)

        if amazon.empty or walmart.empty:
            print("Could not find candidates from both retailers.")
            print("Try a more specific product name or model number.")
            continue

        matches = []

        for _, a in amazon.iterrows():
            for _, w in walmart.iterrows():
                probability, is_match = predict_pair(bundle, a, w)

                if is_match:
                    try:
                        ap = float(
                            normalize(a.get("price", "")).replace(" ", "")
                        )
                    except (ValueError, TypeError):
                        ap = None

                    try:
                        wp = float(
                            normalize(w.get("price", "")).replace(" ", "")
                        )
                    except (ValueError, TypeError):
                        wp = None

                    matches.append({
                        "probability": probability,
                        "amazon_id": a["id"],
                        "amazon_title": a["title"],
                        "amazon_price": a.get("price", ""),
                        "walmart_id": w["id"],
                        "walmart_title": w["title"],
                        "walmart_price": w.get("price", ""),
                        "amazon_numeric_price": ap,
                        "walmart_numeric_price": wp,
                    })

        if not matches:
            print("No cross-retailer matches passed the M6 threshold.")
            print("This does not prove that no equivalent product exists.")
            continue

        matches.sort(key=lambda m: m["probability"], reverse=True)

        print(f"\nMatching pairs found: {len(matches)}")
        for i, m in enumerate(matches[:10], 1):
            print(f"\n{i}. M6 match probability: {m['probability']:.2%}")
            print("Amazon:", m["amazon_title"])
            print("  Recorded price:", m["amazon_price"])
            print("  Product ID:", m["amazon_id"])
            print("Walmart:", m["walmart_title"])
            print("  Recorded price:", m["walmart_price"])
            print("  Product ID:", m["walmart_id"])

            ap = m["amazon_numeric_price"]
            wp = m["walmart_numeric_price"]
            if ap is not None and wp is not None:
                cheaper = "Amazon" if ap < wp else (
                    "Walmart" if wp < ap else "Same recorded price"
                )
                print("Lower numeric price:", cheaper)
                print("Verify currency and bundle specifications before buying.")

if __name__ == "__main__":
    main()
