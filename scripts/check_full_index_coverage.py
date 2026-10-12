
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path(r"D:\DLPROJECT")

INDEX_FILE = ROOT / "data" / "indexes" / "product_text_faiss_us_full.index"
IDS_FILE = ROOT / "data" / "indexes" / "product_text_faiss_us_full_ids.npy"
EXAMPLES_FILE = (
    ROOT / "data" / "raw" / "esci" /
    "shopping_queries_dataset_examples.parquet"
)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading product IDs...")
indexed_ids = np.load(IDS_FILE, allow_pickle=False).astype(str)
indexed_set = set(indexed_ids.tolist())

print("Reading official ESCI test examples...")
df = pd.read_parquet(
    EXAMPLES_FILE,
    columns=[
        "query_id", "product_id", "product_locale",
        "esci_label", "split"
    ],
)

test = df[
    (df["split"] == "test")
    & (df["product_locale"] == "us")
].copy()

# E and S are treated as relevant, consistent with the earlier evaluation.
relevant = test[
    test["esci_label"].isin(["E", "S"])
].copy()

relevant_ids = set(relevant["product_id"].astype(str).unique())
covered_ids = relevant_ids.intersection(indexed_set)

relevant["product_id_str"] = relevant["product_id"].astype(str)
covered_queries = relevant[
    relevant["product_id_str"].isin(indexed_set)
]["query_id"].nunique()

total_queries = test["query_id"].nunique()

print("\n========== FULL INDEX VERIFICATION ==========")
print("FAISS vectors:", index.ntotal)
print("Product IDs:", len(indexed_ids))
print("Unique indexed IDs:", len(indexed_set))
print("Embedding dimension:", index.d)
print("Unique relevant US test products:", len(relevant_ids))
print("Relevant products covered:", len(covered_ids))
print(
    "Relevant product coverage:",
    f"{100 * len(covered_ids) / len(relevant_ids):.2f}%"
)
print("US test queries:", total_queries)
print("Queries with at least one relevant product indexed:", covered_queries)
print(
    "Query coverage:",
    f"{100 * covered_queries / total_queries:.2f}%"
)

assert index.ntotal == len(indexed_ids), "Index/ID count mismatch!"
assert index.d == 384, "Unexpected embedding dimension!"
assert len(indexed_set) == len(indexed_ids), "Duplicate indexed IDs!"
print("\nAll structural checks passed.")
