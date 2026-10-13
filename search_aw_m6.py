
import re
import sys
from pathlib import Path

import pandas as pd

from predict_aw_m6 import load_m6, predict_pair


# =========================================================
# AI PRICE HUNTER — INTEGRATED M6 SEARCH ENGINE
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "downloads"
CATALOG_FILE = DATA_DIR / "amazon_walmart.tsv"

MATCH_THRESHOLD = 0.890
MAX_CANDIDATES = 50


# =========================================================
# TEXT UTILITIES
# =========================================================

def normalize(value):
    if value is None or pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9.+-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_price(value):
    """Parse recorded prices; reject missing or invalid values."""
    if value is None or pd.isna(value):
        return None

    value = str(value).strip().replace(",", "")

    if not value or value.lower() in {
        "nan", "none", "null", "n/a", "na", "-"
    }:
        return None

    match = re.fullmatch(r"\$?\s*(\d+)\s+(\d{2})", value)
    if match:
        price = float(f"{match.group(1)}.{match.group(2)}")
        return price if price > 0 else None

    match = re.fullmatch(r"\$?\s*(\d+(?:\.\d{1,2})?)", value)
    if match:
        price = float(match.group(1))
        return price if price > 0 else None

    return None


# =========================================================
# SPECIFICATIONS
# =========================================================

def get_capacities(text):
    """Extract explicit storage capacities, normalized to MB."""
    text = normalize(text)

    pattern = (
        r"\b(\d+(?:\.\d+)?)\s*"
        r"(tb|terabytes?|gb|gigabytes?|mb|megabytes?)\b"
    )

    result = set()

    for number, unit in re.findall(pattern, text):
        amount = float(number)

        if unit.startswith("tb"):
            amount *= 1024 * 1024
        elif unit.startswith(("gb", "gigabyte")):
            amount *= 1024

        result.add(round(amount, 3))

    return result


def get_vertex_series(text):
    match = re.search(
        r"\bvertex\s*[- ]?\s*(\d+)\b",
        normalize(text),
    )
    return int(match.group(1)) if match else None


# =========================================================
# BRAND AND CATEGORY FILTERING
# =========================================================

KNOWN_BRANDS = [
    "western digital",
    "audio technica",
    "microsoft",
    "sennheiser",
    "logitech",
    "samsung",
    "toshiba",
    "iomega",
    "netgear",
    "buffalo",
    "philips",
    "seagate",
    "sanfisk",  # harmless if absent from the catalog
    "sandisk",
    "sony",
    "asus",
    "apple",
    "acer",
    "jvc",
    "coby",
    "belkin",
    "d link",
    "linksys",
    "panasonic",
    "verbatim",
    "fantom",
]


def query_brand(query):
    """Return a known brand explicitly named in the query."""
    q = normalize(query)

    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(brand) + r"\b", q):
            return brand

    return None


CATEGORY_TERMS = {
    "headphones": [
        "headphone", "headphones", "earphone", "earphones",
        "earbud", "earbuds", "headset", "headsets",
    ],
    "mouse": ["mouse", "mice"],
    "keyboard": ["keyboard", "keyboards"],
    "speaker": ["speaker", "speakers"],
    "laptop": ["laptop", "notebook"],
    "ssd": ["ssd", "solid state drive"],
    "hard drive": [
        "hard drive", "hard disk", "external drive",
        "portable drive", "network attached storage",
        "network storage",
    ],
    "webcam": ["webcam", "web camera"],
    "camcorder": ["camcorder", "video camera"],
}


def query_category(query):
    """Identify a category explicitly mentioned in a query."""
    q = normalize(query)

    # More specific categories should take precedence.
    for category, terms in CATEGORY_TERMS.items():
        if any(term in q for term in terms):
            return category

    return None


def title_has_category(title, category):
    """Check whether a listing title describes the requested category."""
    t = normalize(title)

    if category is None:
        return True

    terms = CATEGORY_TERMS[category]

    if category == "headphones":
        # Exclude standalone speakers even if their title mentions
        # noise cancellation.
        if "speaker" in t and not any(
            word in t for word in ("headphone", "earphone", "headset")
        ):
            return False
        return any(term in t for term in terms)

    if category == "mouse":
        is_mouse = any(
            re.search(r"\b" + re.escape(term) + r"\b", t)
            for term in terms
        )
        if "mouse pad" in t or "mousepad" in t:
            return False
        if "keyboard" in t and (
            "combo" in t or "bundle" in t
        ):
            return False
        return is_mouse

    if category == "ssd":
        return "ssd" in t or "solid state drive" in t

    return any(term in t for term in terms)


def query_spec_compatible(query, title):
    """Reject listings that conflict with explicit query requirements."""
    q = normalize(query)
    t = normalize(title)

    # Enforce a known brand if the user explicitly names one.
    brand = query_brand(q)
    if brand and not re.search(r"\b" + re.escape(brand) + r"\b", t):
        return False

    # Enforce requested storage capacity.
    query_capacities = get_capacities(q)
    title_capacities = get_capacities(t)

    if query_capacities:
        if not title_capacities.intersection(query_capacities):
            return False

    # Enforce Vertex series when specified.
    query_series = get_vertex_series(q)
    title_series = get_vertex_series(t)

    if query_series is not None and title_series != query_series:
        return False

    # Enforce requested category.
    category = query_category(q)
    if category and not title_has_category(t, category):
        return False

    return True


def pair_is_compatible(amazon, walmart):
    """Check obvious conflicts between two listings."""
    a_title = normalize(amazon.get("title", ""))
    w_title = normalize(walmart.get("title", ""))

    a_capacity = get_capacities(a_title)
    w_capacity = get_capacities(w_title)

    if a_capacity and w_capacity:
        if not a_capacity.intersection(w_capacity):
            return False, "Different capacities"

    if bool(a_capacity) != bool(w_capacity):
        return False, "Capacity missing from one listing"

    a_series = get_vertex_series(a_title)
    w_series = get_vertex_series(w_title)

    if a_series is not None and w_series is not None:
        if a_series != w_series:
            return False, "Different Vertex series"

    if "vertex" in a_title or "vertex" in w_title:
        if (a_series is None) != (w_series is None):
            return False, "Vertex series not verified"

    # When both listings identify a known brand, reject conflicts.
    a_brand = query_brand(a_title)
    w_brand = query_brand(w_title)

    if a_brand and w_brand and a_brand != w_brand:
        return False, "Different brands"

    return True, (
        "No obvious specification conflict; "
        "exact model still needs verification"
    )


# =========================================================
# MARKETPLACE NORMALIZATION
# =========================================================

def identify_marketplace(value):
    """Identify marketplace using the dataset column."""
    value = normalize(value)

    if value in {"amazon", "amazon.com", "amazoncom"}:
        return "amazon"

    if value in {"walmart", "wal mart", "walmart.com", "walmartcom"}:
        return "walmart"

    if "walmart" in value or "wal mart" in value:
        return "walmart"

    if "amazon" in value:
        return "amazon"

    return ""


# =========================================================
# DATASET LOADING
# =========================================================

def load_catalog():
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG_FILE}\n"
            "Expected: downloads/amazon_walmart.tsv"
        )

    print(f"\nLoading catalog: {CATALOG_FILE}")

    catalog = pd.read_csv(
        CATALOG_FILE,
        sep="\t",
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )

    catalog.columns = [
        str(column).strip().lower()
        for column in catalog.columns
    ]

    aliases = {
        "product_title": "title",
        "product_name": "title",
        "name": "title",
        "product_price": "price",
        "product_id": "id",
        "model_no": "modelno",
        "model_number": "modelno",
        "retailer": "dataset",
        "marketplace": "dataset",
        "store": "dataset",
    }

    for old, new in aliases.items():
        if new not in catalog.columns and old in catalog.columns:
            catalog.rename(columns={old: new}, inplace=True)

    required = [
        "title", "price", "id", "dataset",
        "brand", "modelno", "shortdescr", "longdescr",
    ]

    for column in required:
        if column not in catalog.columns:
            catalog[column] = ""
        catalog[column] = catalog[column].fillna("").astype(str)

    catalog["dataset"] = catalog["dataset"].apply(
        identify_marketplace
    )

    catalog = catalog[
        catalog["dataset"].isin(["amazon", "walmart"])
        & catalog["title"].str.strip().ne("")
    ].copy()

    catalog["_key"] = (
        catalog["dataset"] + "|"
        + catalog["id"].astype(str) + "|"
        + catalog["title"].map(normalize) + "|"
        + catalog["price"].astype(str)
    )

    catalog = catalog.drop_duplicates("_key").reset_index(drop=True)

    amazon_count = int((catalog["dataset"] == "amazon").sum())
    walmart_count = int((catalog["dataset"] == "walmart").sum())

    print(f"Amazon records: {amazon_count:,}")
    print(f"Walmart records: {walmart_count:,}")

    if amazon_count == 0 or walmart_count == 0:
        raise ValueError(
            "Could not identify both Amazon and Walmart records. "
            "Inspect the dataset marketplace column."
        )

    return catalog


# =========================================================
# CANDIDATE RETRIEVAL
# =========================================================

def retrieve_candidates(query, catalog, marketplace):
    candidates = catalog[
        catalog["dataset"] == marketplace
    ].copy()

    # Strict brand, category, and specification checks first.
    candidates = candidates[
        candidates["title"].apply(
            lambda title: query_spec_compatible(query, title)
        )
    ].copy()

    if candidates.empty:
        return candidates

    query_terms = set(normalize(query).split())

    ignored = {
        "the", "and", "for", "with", "from",
        "solid", "state", "drive", "inch",
    }
    query_terms -= ignored

    def lexical_score(row):
        text = normalize(
            str(row.get("title", "")) + " "
            + str(row.get("brand", "")) + " "
            + str(row.get("modelno", "")) + " "
            + str(row.get("shortdescr", ""))
        )
        return len(query_terms.intersection(set(text.split())))

    candidates["_query_score"] = candidates.apply(
        lexical_score, axis=1
    )

    candidates = candidates[
        candidates["_query_score"] > 0
    ]

    return candidates.sort_values(
        "_query_score",
        ascending=False,
        kind="stable",
    ).head(MAX_CANDIDATES).copy()


# =========================================================
# M6 SEARCH AND PRICE COMPARISON
# =========================================================

def search_products(query, catalog, bundle):
    amazon = retrieve_candidates(query, catalog, "amazon")
    walmart = retrieve_candidates(query, catalog, "walmart")

    print(f"\nAmazon candidates: {len(amazon)}")
    print(f"Walmart candidates: {len(walmart)}")

    if amazon.empty or walmart.empty:
        print("\nNo suitable listings found at both marketplaces.")
        print("No best-price recommendation can be made.")
        return

    print("\nRunning integrated M6 matching...")
    print("Checking categories, specifications, and recorded prices...")

    matches = []
    rejected = 0

    for _, a in amazon.iterrows():
        for _, w in walmart.iterrows():

            if not query_spec_compatible(query, a["title"]):
                rejected += 1
                continue

            if not query_spec_compatible(query, w["title"]):
                rejected += 1
                continue

            compatible, reason = pair_is_compatible(a, w)

            if not compatible:
                rejected += 1
                continue

            amazon_price = parse_price(a.get("price"))
            walmart_price = parse_price(w.get("price"))

            if amazon_price is None or walmart_price is None:
                continue

            try:
                probability, is_match = predict_pair(bundle, a, w)
                probability = float(probability)
            except Exception as exc:
                print(f"Warning: M6 prediction failed: {exc}")
                continue

            if not is_match or probability < MATCH_THRESHOLD:
                continue

            matches.append({
                "probability": probability,
                "amazon": a,
                "walmart": w,
                "amazon_price": amazon_price,
                "walmart_price": walmart_price,
                "reason": reason,
                "best_price": min(amazon_price, walmart_price),
            })

    print(f"Pairs rejected by compatibility checks: {rejected}")

    if not matches:
        print("\nNo pairs passed M6, specification, and price checks.")
        print("This does not prove that no matching product exists.")
        print("No reliable deal can be recommended from these candidates.")
        return

    matches.sort(
        key=lambda item: (
            item["best_price"],
            -item["probability"],
        )
    )

    print(f"Valid priced comparisons: {len(matches)}")

    for index, match in enumerate(matches[:10], start=1):
        a = match["amazon"]
        w = match["walmart"]
        ap = match["amazon_price"]
        wp = match["walmart_price"]

        print("\n" + "=" * 65)
        print(
            f"Match {index} | "
            f"M6 probability: {match['probability']:.2%}"
        )

        print("\nAmazon")
        print("  Product:", a["title"])
        print("  ID:", a["id"])
        print(f"  Recorded price: {ap:.2f}")

        print("\nWalmart")
        print("  Product:", w["title"])
        print("  ID:", w["id"])
        print(f"  Recorded price: {wp:.2f}")

        if ap < wp:
            print(f"\nLower recorded price: Amazon ({wp - ap:.2f} less)")
        elif wp < ap:
            print(f"\nLower recorded price: Walmart ({ap - wp:.2f} less)")
        else:
            print("\nBoth listings have the same recorded price.")

        print("Compatibility:", match["reason"])
        print("Currency, condition, shipping, and bundle contents are unverified.")

    best = matches[0]

    if best["amazon_price"] <= best["walmart_price"]:
        retailer = "Amazon"
        product = best["amazon"]["title"]
        price = best["amazon_price"]
    else:
        retailer = "Walmart"
        product = best["walmart"]["title"]
        price = best["walmart_price"]

    print("\n" + "=" * 65)
    print("BEST RECORDED DEAL CANDIDATE")
    print("Retailer:", retailer)
    print("Product:", product)
    print(f"Recorded price: {price:.2f}")
    print("Dataset-based result only; this is not a live offer.")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 60)
    print("             AI PRICE HUNTER — M6")
    print("=" * 60)
    print("Integrated engine: adaptive_text_attribute_router")
    print(f"Match threshold: {MATCH_THRESHOLD:.3f}")
    print("Marketplaces: Amazon and Walmart")
    print("Prices: recorded dataset values, not live offers")
    print("Enter q to quit.")

    try:
        catalog = load_catalog()
        print(f"Total usable records: {len(catalog):,}")
        bundle, model_products = load_m6()
    except Exception as exc:
        print(f"\nStartup error: {exc}")
        sys.exit(1)

    while True:
        try:
            query = input(
                "\nEnter product name (or q to quit): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting safely.")
            break

        if query.lower() in {"q", "quit", "exit"}:
            print("Exiting AI Price Hunter.")
            break

        if not query:
            print("Please enter a product name.")
            continue

        try:
            search_products(query, catalog, bundle)
        except KeyboardInterrupt:
            print("\nSearch interrupted.")
        except Exception as exc:
            print(f"\nSearch error: {exc}")


if __name__ == "__main__":
    main()
