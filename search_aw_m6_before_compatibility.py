
from pathlib import Path
import sys
import re
import pandas as pd

ROOT = Path(r"D:\DLPROJECT")
DATA_DIR = ROOT / "data" / "processed" / "amazon_walmart_m6"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

from predict_aw_m6 import load_m6, predict_pair


def normalize(value):
    """Normalize text for product search."""
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+", " ",
        re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    ).strip()


def parse_recorded_price(value):
    """
    Parse the dataset's observed price format.

    Examples:
        '12 65' -> 12.65
        '6 99'  -> 6.99
        '24'    -> 24.00

    Returns None for unrecognized values.
    Currency is not inferred.
    """
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    # The source dataset uses spaces instead of decimal points.
    match = re.fullmatch(r"(\d+)(?:\s+(\d{2}))?", value)

    if not match:
        return None

    whole = int(match.group(1))
    fractional = int(match.group(2) or "00")

    return whole + fractional / 100


def prepare_catalog(products):
    """Prepare searchable fields without changing the source catalog."""
    products = products.copy()

    for col in [
        "title", "brand", "modelno", "shortdescr",
        "longdescr", "techdetails", "price", "dataset"
    ]:
        if col not in products.columns:
            products[col] = ""

    products["dataset"] = (
        products["dataset"].fillna("").astype(str).str.lower().str.strip()
    )

    products = products[
        products["dataset"].isin(["amazon", "walmart"])
    ].copy()

    products["search_text"] = (
        products["title"].fillna("").astype(str) + " "
        + products["brand"].fillna("").astype(str) + " "
        + products["modelno"].fillna("").astype(str) + " "
        + products["shortdescr"].fillna("").astype(str) + " "
        + products["longdescr"].fillna("").astype(str)
    ).map(normalize)

    return products


def retrieve_candidates(products, query, retailer, limit=15):
    """Retrieve candidates using lexical overlap with the query."""
    query_terms = set(normalize(query).split())

    if not query_terms:
        return products.iloc[0:0].copy()

    subset = products[products["dataset"] == retailer].copy()

    def score(text):
        product_terms = set(text.split())
        return len(query_terms & product_terms) / len(query_terms)

    subset["_query_score"] = subset["search_text"].map(score)

    subset = subset[subset["_query_score"] > 0]
    return subset.sort_values(
        "_query_score", ascending=False
    ).head(limit)


def print_recorded_price(label, value):
    parsed = parse_recorded_price(value)

    if parsed is None:
        print(f"{label}: unavailable (unrecognized recorded price)")
        return None

    print(f"{label}: {parsed:.2f} (recorded price; currency unverified)")
    return parsed


def search_products(bundle, products, query):
    amazon = retrieve_candidates(products, query, "amazon")
    walmart = retrieve_candidates(products, query, "walmart")

    if amazon.empty:
        print("No Amazon candidates found. Try another query.")
        return

    if walmart.empty:
        print("No Walmart candidates found. Try another query.")
        return

    print(
        f"\nCandidates: {len(amazon)} Amazon, "
        f"{len(walmart)} Walmart"
    )
    print("Checking candidate pairs with M6...")

    matches = []

    for _, a in amazon.iterrows():
        for _, w in walmart.iterrows():
            probability, is_match = predict_pair(bundle, a, w)

            if not is_match:
                continue

            matches.append({
                "probability": probability,
                "amazon_id": str(a["id"]),
                "amazon_title": str(a.get("title", "")),
                "amazon_price_raw": a.get("price", ""),
                "amazon_price": parse_recorded_price(a.get("price", "")),
                "walmart_id": str(w["id"]),
                "walmart_title": str(w.get("title", "")),
                "walmart_price_raw": w.get("price", ""),
                "walmart_price": parse_recorded_price(w.get("price", "")),
            })

    if not matches:
        print("\nNo candidate pair passed the M6 threshold.")
        print("This does not prove that no equivalent listing exists.")
        return

    # Show strongest predicted matches first.
    matches.sort(key=lambda x: x["probability"], reverse=True)

    print(f"\nMatching pairs found: {len(matches)}")

    for index, match in enumerate(matches[:10], start=1):
        print(f"\n{'=' * 65}")
        print(f"Match {index}")
        print(f"M6 probability: {match['probability']:.2%}")

        print("\nAmazon")
        print("  Title:", match["amazon_title"])
        print("  ID:", match["amazon_id"])
        ap = print_recorded_price(
            "  Price", match["amazon_price_raw"]
        )

        print("\nWalmart")
        print("  Title:", match["walmart_title"])
        print("  ID:", match["walmart_id"])
        wp = print_recorded_price(
            "  Price", match["walmart_price_raw"]
        )

        if ap is None or wp is None:
            print("\nRecommendation: unavailable; check both prices.")
        elif ap < wp:
            print(f"\nLower recorded price: Amazon ({wp - ap:.2f} less)")
        elif wp < ap:
            print(f"\nLower recorded price: Walmart ({ap - wp:.2f} less)")
        else:
            print("\nBoth listings have the same recorded price.")

        print(
            "Reminder: confirm currency, condition, shipping, "
            "and bundle specifications before comparing offers."
        )


def main():
    print("\n" + "=" * 55)
    print("          AI PRICE HUNTER — M6")
    print("=" * 55)

    try:
        bundle, catalog = load_m6()
        products = prepare_catalog(catalog)

        print("Official model:", bundle["winner_name"])
        print(f"Match threshold: {bundle['threshold']:.3f}")
        print("Retailers: Amazon and Walmart")
        print("Prices: recorded dataset values, not live offers")
        print("\nEnter a product name or model number.")
        print("Type q to quit.")

        while True:
            try:
                query = input("\nProduct to search: ").strip()
            except EOFError:
                print("\nInput ended. Exiting.")
                break

            if query.lower() in {"q", "quit", "exit"}:
                print("Exiting AI Price Hunter.")
                break

            if len(query) < 2:
                print("Enter at least two characters.")
                continue

            search_products(bundle, products, query)

    except KeyboardInterrupt:
        print("\n\nSearch interrupted. Exiting safely.")
    except FileNotFoundError as exc:
        print("\nRequired project file not found:", exc)
        print("Check that the model and catalog files exist.")
    except Exception as exc:
        print(f"\nUnexpected error: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
