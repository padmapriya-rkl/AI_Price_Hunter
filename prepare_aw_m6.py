
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "downloads" / "amazon_walmart.tsv"
OUTPUT_DIR = ROOT / "data" / "processed" / "amazon_walmart_m6"
OUTPUT_FILE = OUTPUT_DIR / "products.parquet"
REPORT_FILE = OUTPUT_DIR / "data_report.txt"


def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    print("Loading Amazon-Walmart dataset...")
    df = pd.read_csv(
        SOURCE,
        sep="\t",
        dtype=str,
        low_memory=False,
        on_bad_lines="warn",
    )

    required = {"dataset", "title", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in [
        "brand", "modelno", "shortdescr",
        "longdescr", "techdetails", "imageurl", "id"
    ]:
        if col not in df.columns:
            df[col] = ""

    df["marketplace"] = (
        df["dataset"].fillna("").str.strip().str.lower()
    )
    df = df[df["marketplace"].isin(["amazon", "walmart"])].copy()

    df["title"] = df["title"].fillna("").str.strip()
    df["brand"] = df["brand"].fillna("").str.strip()
    df["modelno"] = df["modelno"].fillna("").str.strip()
    df["imageurl"] = df["imageurl"].fillna("").str.strip()

    df["title_norm"] = df["title"].map(normalize)
    df["brand_norm"] = df["brand"].map(normalize)
    df["model_norm"] = df["modelno"].map(normalize)

    df["product_text"] = (
        df["title"].fillna("") + " " +
        df["shortdescr"].fillna("") + " " +
        df["longdescr"].fillna("") + " " +
        df["techdetails"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    df["has_image_url"] = df["imageurl"].str.match(
        r"^https?://", case=False, na=False
    )

    # Retain source identifiers; never treat them as cross-marketplace IDs.
    df["listing_id"] = (
        df["marketplace"] + "_" + df["id"].fillna("").astype(str)
    )

    df = df[df["title_norm"].ne("")].copy()
    df = df.drop_duplicates(
        subset=["marketplace", "title_norm", "price"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    report = [
        "AMAZON-WALMART M6 DATA REPORT",
        "=" * 40,
        f"Listings retained: {len(df):,}",
        f"Amazon listings: {(df['marketplace'] == 'amazon').sum():,}",
        f"Walmart listings: {(df['marketplace'] == 'walmart').sum():,}",
        f"Rows with URL-shaped image links: {df['has_image_url'].sum():,}",
        f"Rows without URL-shaped image links: {(~df['has_image_url']).sum():,}",
        f"Unique brands: {df['brand_norm'].nunique():,}",
        f"Non-empty model numbers: {df['model_norm'].ne('').sum():,}",
        "",
        "Note: URL format does not prove that an image is reachable.",
        "Prices are source values; currency and current availability",
        "must be verified separately.",
    ]

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nSaved products: {OUTPUT_FILE}")
    print(f"Saved report:   {REPORT_FILE}")


if __name__ == "__main__":
    main()
