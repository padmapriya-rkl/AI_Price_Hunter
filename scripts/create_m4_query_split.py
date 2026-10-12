import os
import numpy as np
import pyarrow.parquet as pq


TEST_FILE = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\sqid_esci_test.parquet"
)

OUTPUT_DIR = (
    r"D:\DLPROJECT\data\processed\multimodal"
    r"\m4"
)

RANDOM_SEED = 42


print("=" * 70)
print("M4 QUERY-LEVEL SPLIT")
print("=" * 70)

print("\nReading query IDs...")

parquet_file = pq.ParquetFile(TEST_FILE)

query_ids = set()

for batch in parquet_file.iter_batches(
    batch_size=50000,
    columns=["query_id"]
):
    values = batch.column("query_id").to_pylist()

    for query_id in values:
        if query_id is not None:
            query_ids.add(str(query_id))


query_ids = np.array(
    sorted(query_ids),
    dtype=str
)

print(
    "Total unique queries:",
    len(query_ids)
)


rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(query_ids)


n = len(query_ids)

n_train = int(0.70 * n)
n_val = int(0.15 * n)

train_ids = query_ids[:n_train]

val_ids = query_ids[
    n_train:n_train + n_val
]

test_ids = query_ids[
    n_train + n_val:
]


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_query_ids.npy"
    ),
    train_ids
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_query_ids.npy"
    ),
    val_ids
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_query_ids.npy"
    ),
    test_ids
)


print("\n" + "=" * 70)
print("M4 SPLIT CREATED")
print("=" * 70)

print("Train queries:", len(train_ids))
print("Validation queries:", len(val_ids))
print("Test queries:", len(test_ids))

print(
    "Total:",
    len(train_ids)
    + len(val_ids)
    + len(test_ids)
)

print("\nSaved to:")
print(OUTPUT_DIR)

print("=" * 70)