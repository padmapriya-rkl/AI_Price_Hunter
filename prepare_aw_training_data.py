
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(r"D:\DLPROJECT")
DOWNLOADS = ROOT / "downloads"
OUTPUT = ROOT / "data" / "processed" / "amazon_walmart_m6"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Load the product catalog
products = pd.read_csv(
    DOWNLOADS / "amazon_walmart.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

products["id"] = products["id"].str.strip()
products["dataset"] = products["dataset"].str.strip().str.lower()
products = products.drop_duplicates(subset="id", keep="first")

# Keep the fields needed for product matching
required = [
    "id", "dataset", "title", "brand", "modelno",
    "category", "price", "shortdescr", "longdescr",
    "techdetails", "dimensions", "imageurl"
]
for col in required:
    if col not in products.columns:
        products[col] = ""

products = products.set_index("id", drop=False)

print("Unique product IDs:", len(products))
print("Retailer counts:")
print(products["dataset"].value_counts().to_string())


def load_pairs(filename, label):
    pairs = pd.read_csv(
        DOWNLOADS / filename,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    pairs["id1"] = pairs["id1"].str.strip()
    pairs["id2"] = pairs["id2"].str.strip()

    pairs = pairs[["id1", "id2"]].copy()
    pairs["label"] = label
    return pairs


# DPL = matching products; NDPL = non-matching products
positive = load_pairs("amazon_walmart_DPL.tsv", 1)
negative = load_pairs("amazon_walmart_NDPL.tsv", 0)

pairs = pd.concat([positive, negative], ignore_index=True)

# Remove missing IDs and pairs that cannot be mapped to products
valid_ids = (
    pairs["id1"].isin(products.index)
    & pairs["id2"].isin(products.index)
)
invalid_count = int((~valid_ids).sum())
pairs = pairs[valid_ids].copy()

# Remove self-pairs and duplicate pairs, regardless of ID order
pairs = pairs[pairs["id1"] != pairs["id2"]].copy()
pairs["pair_key"] = pairs.apply(
    lambda r: tuple(sorted((r["id1"], r["id2"]))),
    axis=1,
)
pairs = pairs.drop_duplicates("pair_key").drop(columns="pair_key")

# Require one Amazon listing and one Walmart listing per pair
cross_marketplace = []
for row in pairs.itertuples(index=False):
    retailer1 = products.at[row.id1, "dataset"]
    retailer2 = products.at[row.id2, "dataset"]
    cross_marketplace.append(
        {retailer1, retailer2} == {"amazon", "walmart"}
    )

pairs = pairs[pd.Series(cross_marketplace, index=pairs.index)].copy()

# Attach product details to each pair
def add_product_fields(frame, side):
    details = products.loc[frame[f"id{side}"]].reset_index(drop=True)
    details = details.add_prefix(f"p{side}_")
    return details

pairs = pairs.reset_index(drop=True)
left = add_product_fields(pairs, "1")
right = add_product_fields(pairs, "2")

dataset = pd.concat(
    [pairs[["id1", "id2", "label"]], left, right],
    axis=1,
)

# Shuffle and create reproducible pair-level splits
train, remainder = train_test_split(
    dataset,
    test_size=0.30,
    random_state=42,
    stratify=dataset["label"],
)

validation, test = train_test_split(
    remainder,
    test_size=0.50,
    random_state=42,
    stratify=remainder["label"],
)

train.to_parquet(OUTPUT / "train_pairs.parquet", index=False)
validation.to_parquet(OUTPUT / "validation_pairs.parquet", index=False)
test.to_parquet(OUTPUT / "test_pairs.parquet", index=False)
dataset.to_csv(OUTPUT / "all_labeled_pairs.csv", index=False)

report = [
    "Amazon-Walmart M6 training data report",
    f"Unique catalog IDs: {len(products)}",
    f"Invalid-ID pairs removed: {invalid_count}",
    f"Usable pairs: {len(dataset)}",
    f"Positive pairs: {(dataset['label'] == 1).sum()}",
    f"Negative pairs: {(dataset['label'] == 0).sum()}",
    f"Training pairs: {len(train)}",
    f"Validation pairs: {len(validation)}",
    f"Test pairs: {len(test)}",
    "",
    "Label 1 = DPL (duplicate/matching pair)",
    "Label 0 = NDPL (non-duplicate pair)",
    "Splits are pair-level; product IDs may occur in multiple splits.",
]

(OUTPUT / "training_data_report.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n".join(report))
print(f"\nFiles saved to: {OUTPUT}")
