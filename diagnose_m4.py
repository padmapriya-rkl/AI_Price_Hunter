import pandas as pd
import numpy as np

PATH = r"D:\DLPROJECT\results\m4_adaptive_text_image_per_query.csv"

df = pd.read_csv(PATH)

print("=" * 60)
print("M4 ROUTER DIAGNOSTIC")
print("=" * 60)

print(f"Total queries: {len(df)}")

# ------------------------------------------------------------
# 1. Overall
# ------------------------------------------------------------

print("\nOverall:")
print(f"Recall@10 : {df['recall_at_10'].mean():.4f}")
print(f"nDCG@10   : {df['ndcg_at_10'].mean():.4f}")

# ------------------------------------------------------------
# 2. Router groups
# ------------------------------------------------------------

df["router_group"] = np.where(
    df["alpha_text"] > 0.5,
    "TEXT-DOMINANT",
    "IMAGE-DOMINANT"
)

print("\nRouter groups:")
print(df["router_group"].value_counts())

print("\nPerformance by router group:")

grouped = df.groupby("router_group")[
    ["alpha_text", "alpha_image",
     "recall_at_10", "ndcg_at_10"]
].mean()

print(grouped)

# ------------------------------------------------------------
# 3. Weight ranges
# ------------------------------------------------------------

print("\nRouting weights:")

print(
    f"Mean alpha_text  : {df['alpha_text'].mean():.4f}"
)

print(
    f"Mean alpha_image : {df['alpha_image'].mean():.4f}"
)

print(
    f"Min alpha_text   : {df['alpha_text'].min():.4f}"
)

print(
    f"Max alpha_text   : {df['alpha_text'].max():.4f}"
)

# ------------------------------------------------------------
# 4. Weight bins
# ------------------------------------------------------------

df["text_weight_bin"] = pd.cut(
    df["alpha_text"],
    bins=[
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ],
    include_lowest=True
)

print("\nPerformance by text-weight range:")

print(
    df.groupby(
        "text_weight_bin",
        observed=True
    )[[
        "recall_at_10",
        "ndcg_at_10",
        "alpha_text",
        "alpha_image"
    ]].agg([
        "mean",
        "count"
    ])
)

# ------------------------------------------------------------
# 5. Correlation
# ------------------------------------------------------------

print("\nCorrelation:")

print(
    "alpha_text vs Recall:",
    df["alpha_text"].corr(
        df["recall_at_10"]
    )
)

print(
    "alpha_text vs nDCG:",
    df["alpha_text"].corr(
        df["ndcg_at_10"]
    )
)

# ------------------------------------------------------------
# 6. Best and worst queries
# ------------------------------------------------------------

print("\nHighest nDCG queries:")

print(
    df.nlargest(10, "ndcg_at_10")[
        [
            "query_id",
            "query",
            "alpha_text",
            "alpha_image",
            "recall_at_10",
            "ndcg_at_10"
        ]
    ].to_string(index=False)
)

print("\nLowest nDCG queries:")

print(
    df.nsmallest(10, "ndcg_at_10")[
        [
            "query_id",
            "query",
            "alpha_text",
            "alpha_image",
            "recall_at_10",
            "ndcg_at_10"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)