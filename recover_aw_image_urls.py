
from pathlib import Path
from urllib.parse import urlparse
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "downloads" / "amazon_walmart.tsv"
OUTPUT_DIR = ROOT / "data" / "processed" / "amazon_walmart_m6"
OUTPUT_FILE = OUTPUT_DIR / "products_with_image_urls.parquet"


def recover_url(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    # The source appears to have had punctuation replaced by spaces.
    # This reconstruction is only a candidate, not a verified URL.
    compact = re.sub(r"\s+", "", value)

    if compact.lower().startswith("http"):
        compact = compact.replace("http:", "http://", 1)
        if compact.lower().startswith("http//"):
            compact = "http://" + compact[6:]
        elif compact.lower().startswith("https//"):
            compact = "https://" + compact[7:]

    parsed = urlparse(compact)

    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return compact

    return ""


def main():
    df = pd.read_csv(
        SOURCE,
        sep="\t",
        dtype=str,
        low_memory=False,
        on_bad_lines="warn",
    )

    if "imageurl" not in df.columns:
        raise ValueError("The source dataset has no imageurl column.")

    df["image_url_candidate"] = df["imageurl"].map(recover_url)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print("Rows:", len(df))
    print("Candidate URLs:", df["image_url_candidate"].ne("").sum())
    print("\nExamples:")

    for value in df.loc[
        df["image_url_candidate"].ne(""),
        "image_url_candidate"
    ].head(10):
        print(value)

    print("\nSaved:", OUTPUT_FILE)
    print("These URLs are candidates; reachability is not yet verified.")


if __name__ == "__main__":
    main()
