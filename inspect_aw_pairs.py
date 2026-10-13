
from pathlib import Path
import pandas as pd

root = Path(r"D:\DLPROJECT\downloads")

products = pd.read_csv(
    root / "amazon_walmart.tsv",
    sep="\t",
    dtype=str,
    usecols=["id", "dataset", "title", "brand", "modelno", "price"],
)

products["id"] = products["id"].str.strip()
products = products.drop_duplicates("id").set_index("id")

dpl = pd.read_csv(
    root / "amazon_walmart_DPL.tsv", sep="\t", dtype=str
)
ndpl = pd.read_csv(
    root / "amazon_walmart_NDPL.tsv", sep="\t", dtype=str
)

print("Products:", len(products))
print("DPL pairs:", len(dpl))
print("NDPL pairs:", len(ndpl))


def inspect_pairs(frame, label):
    frame = frame.copy()
    frame["id1"] = frame["id1"].str.strip()
    frame["id2"] = frame["id2"].str.strip()

    valid = (
        frame["id1"].isin(products.index)
        & frame["id2"].isin(products.index)
    )

    print(f"\n{label}: valid pairs = {valid.sum()}/{len(frame)}")

    for _, pair in frame[valid].head(5).iterrows():
        a = products.loc[pair["id1"]]
        b = products.loc[pair["id2"]]

        print(f"\nPair: {pair['id1']} <-> {pair['id2']}")
        print("A:", a["dataset"], "|", a["title"])
        print("B:", b["dataset"], "|", b["title"])
        print("Brands:", a["brand"], "/", b["brand"])
        print("Models:", a["modelno"], "/", b["modelno"])

        if "class" in frame.columns:
            print("Class:", pair["class"])


inspect_pairs(dpl, "DPL")
inspect_pairs(ndpl, "NDPL")
