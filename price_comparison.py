
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# AI PRICE HUNTER
# Amazon vs Walmart | Terminal application
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "downloads" / "amazon_walmart.tsv"

MAX_RESULTS = 5
MIN_MATCH_SCORE = 65

ACCESSORY_TERMS = {
    "case", "cover", "charger", "cable", "adapter", "mount",
    "holder", "screen protector", "tempered glass", "protector",
    "pouch", "battery", "headphone", "headphones", "earphone",
    "earphones", "headset", "stand", "dock", "skin", "sleeve",
    "strap", "replacement", "spare part", "panel", "keyboard",
    "stylus", "cooling pad", "carrying bag", "accessory", "accessories"
}


# ============================================================
# UTILITIES
# ============================================================

def normalize(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(
        r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    ).strip()


def has_phrase(text, phrase):
    text = normalize(text)
    phrase = normalize(phrase)
    return bool(phrase) and f" {phrase} " in f" {text} "


def parse_price(value):
    """
    Parse numeric prices and common space-separated decimal formats.

    Examples:
        '12 65'  -> 12.65
        '0 75'   -> 0.75
        '24'     -> 24.00
        '$14.26' -> 14.26

    A value such as '1 500' is ambiguous, so this function
    interprets a final two-digit group as cents only when the
    input has multiple whitespace-separated numeric groups.
    """
    if value is None or pd.isna(value):
        return None

    raw = str(value).strip()

    if not raw or raw.lower() in {
        "nan", "none", "null", "n/a", "na", "-"
    }:
        return None

    # Remove currency symbols but retain decimal points and spaces.
    cleaned = re.sub(r"[^0-9.\s]", "", raw).strip()

    if not cleaned:
        return None

    # Standard decimal price, such as 14.26 or 1,299.50.
    if "." in cleaned:
        try:
            number = float(cleaned.replace(" ", ""))
            return number if number > 0 else None
        except ValueError:
            return None

    groups = cleaned.split()

    try:
        if len(groups) == 1:
            number = float(groups[0])
        elif len(groups[-1]) == 2 and all(g.isdigit() for g in groups):
            # Example: '12 65' means 12.65.
            whole = "".join(groups[:-1])
            number = float(f"{whole}.{groups[-1]}")
        else:
            # Do not guess how an ambiguous value was formatted.
            return None

        return number if number > 0 else None

    except (ValueError, TypeError):
        return None


def looks_like_accessory(title):
    return any(
        has_phrase(title, term)
        for term in ACCESSORY_TERMS
    )


def get_model_tokens(row):
    """
    Extract model identifiers from the available model-number field.
    Product IDs are not assumed to be model numbers.
    """
    model = normalize(row.get("modelno", ""))
    return set(model.split())


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find dataset: {DATA_FILE}"
        )

    print("Loading Amazon/Walmart dataset...")

    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        dtype=str,
        on_bad_lines="warn",
        low_memory=False
    )

    required = {"dataset", "title", "price"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required TSV columns: {sorted(missing)}"
        )

    for column in ("brand", "modelno"):
        if column not in df.columns:
            df[column] = ""

    df["marketplace"] = (
        df["dataset"].fillna("").str.strip().str.lower()
    )

    df = df[df["marketplace"].isin({"amazon", "walmart"})].copy()

    df["title"] = df["title"].fillna("").str.strip()
    df["brand"] = df["brand"].fillna("").str.strip()
    df["modelno"] = df["modelno"].fillna("").str.strip()
    df["title_clean"] = df["title"].map(normalize)
    df["brand_clean"] = df["brand"].map(normalize)
    df["model_clean"] = df["modelno"].map(normalize)
    df["price_numeric"] = df["price"].map(parse_price)

    invalid_prices = int(df["price_numeric"].isna().sum())

    df = df[
        df["title_clean"].ne("") &
        df["price_numeric"].notna()
    ].copy()

    df = df.drop_duplicates(
        subset=["marketplace", "title_clean", "price_numeric"]
    ).reset_index(drop=True)

    print(f"Amazon/Walmart listings loaded: {len(df):,}")
    print(f"Rows excluded for missing/unparsed prices: {invalid_prices:,}")
    print("Price currency: as recorded in the source dataset")
    print("Live price scraping: not enabled")

    return df


# ============================================================
# PRODUCT SEARCH
# ============================================================

def search_products(df, query):
    query_clean = normalize(query)
    query_tokens = set(query_clean.split())

    if not query_tokens:
        return pd.DataFrame()

    query_is_accessory = looks_like_accessory(query)

    candidates = []

    for _, row in df.iterrows():
        title = row["title_clean"]
        title_tokens = set(title.split())

        # All query words must be present as complete tokens.
        if not query_tokens.issubset(title_tokens):
            continue

        # Avoid showing obvious accessories for a main-product query.
        if not query_is_accessory and looks_like_accessory(title):
            continue

        score = fuzz.WRatio(query_clean, title)

        if score < MIN_MATCH_SCORE:
            continue

        item = row.to_dict()
        item["match_score"] = round(score, 1)
        candidates.append(item)

    if not candidates:
        return pd.DataFrame()

    results = pd.DataFrame(candidates)

    # Keep the strongest title matches for each marketplace.
    results = results.sort_values(
        "match_score", ascending=False
    ).drop_duplicates(
        subset=["marketplace", "title_clean"]
    )

    results = results.groupby(
        "marketplace", group_keys=False
    ).head(MAX_RESULTS)

    return results.reset_index(drop=True)


# ============================================================
# CROSS-MARKETPLACE EQUIVALENCE
# ============================================================

def same_product_enough_for_comparison(amazon_row, walmart_row):
    """
    Conservative check. Title similarity alone is not sufficient.
    Require near-identical normalized titles or matching explicit
    model identifiers, plus matching brand when both are present.
    """
    amazon_brand = amazon_row["brand_clean"]
    walmart_brand = walmart_row["brand_clean"]

    if (
        amazon_brand
        and walmart_brand
        and amazon_brand != walmart_brand
    ):
        return False

    title_similarity = fuzz.ratio(
        amazon_row["title_clean"],
        walmart_row["title_clean"]
    )

    amazon_models = get_model_tokens(amazon_row)
    walmart_models = get_model_tokens(walmart_row)

    shared_model = bool(
        amazon_models and walmart_models
        and amazon_models.intersection(walmart_models)
    )

    return title_similarity >= 92 or shared_model


# ============================================================
# DISPLAY
# ============================================================

def display_results(query, results):
    print("\n" + "=" * 76)
    print("             AI PRICE HUNTER")
    print("         AMAZON vs WALMART")
    print("=" * 76)
    print(f"Search: {query}")
    print("Prices are recorded dataset values, not live offers.")
    print("-" * 76)

    if results.empty:
        print("No credible candidates found.")
        print("Try a more specific product name or model number.")
        print("=" * 76)
        return

    for marketplace in ("amazon", "walmart"):
        group = results[
            results["marketplace"] == marketplace
        ].sort_values("price_numeric")

        print(f"\n{marketplace.upper()}")

        if group.empty:
            print("  No matching listings found.")
            continue

        for _, row in group.iterrows():
            print(f"\n  Product: {row['title']}")

            if row["brand"]:
                print(f"  Brand: {row['brand']}")

            if row["modelno"]:
                print(f"  Model number: {row['modelno']}")

            print(f"  Recorded price: {row['price_numeric']:.2f}")
            print(f"  Title similarity: {row['match_score']:.1f}/100")

    print("\n" + "=" * 76)
    print("PRICE RECOMMENDATION")
    print("=" * 76)

    amazon_rows = results[
        results["marketplace"] == "amazon"
    ]
    walmart_rows = results[
        results["marketplace"] == "walmart"
    ]

    verified_pairs = []

    for _, amazon_row in amazon_rows.iterrows():
        for _, walmart_row in walmart_rows.iterrows():
            if same_product_enough_for_comparison(
                amazon_row, walmart_row
            ):
                verified_pairs.append((amazon_row, walmart_row))

    if not verified_pairs:
        print("No sufficiently verified Amazon/Walmart product pair found.")
        print("The application will not invent a cheapest-price winner.")
        print("Inspect model numbers and specifications to verify equivalence.")
    else:
        # Use the cheapest pair among the candidates that pass the check.
        amazon_row, walmart_row = min(
            verified_pairs,
            key=lambda pair: min(
                pair[0]["price_numeric"],
                pair[1]["price_numeric"]
            )
        )

        if amazon_row["price_numeric"] < walmart_row["price_numeric"]:
            winner = amazon_row
            other = walmart_row
            marketplace = "Amazon"
        elif walmart_row["price_numeric"] < amazon_row["price_numeric"]:
            winner = walmart_row
            other = amazon_row
            marketplace = "Walmart"
        else:
            winner = amazon_row
            other = walmart_row
            marketplace = "Amazon and Walmart (same recorded price)"

        print(f"Best candidate: {marketplace}")
        print(f"Product: {winner['title']}")
        print(f"Recorded price: {winner['price_numeric']:.2f}")
        print(f"Other marketplace price: {other['price_numeric']:.2f}")
        print("Confirm the original listings before purchasing.")

    print("\nCurrency caveat:")
    print("The source file's currency is not guaranteed by this script.")
    print("Verify the source dataset's currency before interpreting prices.")
    print("=" * 76)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nStarting AI Price Hunter...\n")

    try:
        df = load_data()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Could not start application: {error}")
        return

    while True:
        try:
            query = input(
                "\nProduct search (or 0 to exit): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AI Price Hunter.")
            break

        if query == "0":
            print("Exiting AI Price Hunter.")
            break

        if not query:
            print("Please enter a product name.")
            continue

        results = search_products(df, query)
        display_results(query, results)


if __name__ == "__main__":
    main()
