import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import load_npz
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"D:\DLPROJECT"
DATA = os.path.join(BASE, "data", "processed", "multimodal")
M4 = os.path.join(DATA, "m4")
M5 = os.path.join(DATA, "m5")
PARQUET = os.path.join(DATA, "sqid_esci_test.parquet")
TRAIN_IDS = os.path.join(M4, "train_query_ids.npy")
VAL_IDS = os.path.join(M4, "val_query_ids.npy")
ATTR_MATRIX = os.path.join(M5, "attribute_tfidf.npz")
ATTR_IDS = os.path.join(M5, "attribute_product_ids.npy")
VECTORIZER = os.path.join(M5, "attribute_vectorizer.pkl")
CHECKPOINT = os.path.join(M5, "m6_adaptive_router.pt")
RESULTS = os.path.join(BASE, "results")
DIM, BATCH_SIZE = 768, 5000
EPOCHS, PATIENCE, LR, MARGIN, SEED = 40, 7, 1e-3, 0.10, 42
POS_LABELS, NEG_LABELS = {"E", "S"}, {"I", "C"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vec(x):
    if x is None:
        return None
    try:
        x = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if x.shape != (DIM,) or not np.isfinite(x).all():
        return None
    n = float(np.linalg.norm(x))
    return None if n < 1e-12 else x / n

class AdaptiveRouter3(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(DIM, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, q):
        return torch.softmax(self.network(q), dim=1)

def load_pairs(ids, name, vectorizer, attr_matrix, attr_rows):
    ids = set(ids)
    cols = ["query_id", "query", "product_id", "product_locale", "esci_label",
            "sqid_query_clip_text_features", "sqid_product_clip_text_features",
            "sqid_product_clip_image_features"]
    records = {}
    parquet = pq.ParquetFile(PARQUET)
    print(f"\nReading {name} rows...")
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        df = batch.to_pandas()
        df = df[df["query_id"].isin(ids)]
        for r in df.itertuples(index=False):
            if str(r.product_locale).strip().lower() != "us":
                continue
            label = str(r.esci_label).strip()
            if label not in POS_LABELS | NEG_LABELS:
                continue
            q = vec(r.sqid_query_clip_text_features)
            t = vec(r.sqid_product_clip_text_features)
            im = vec(r.sqid_product_clip_image_features)
            if q is None or t is None or im is None:
                continue
            e = records.setdefault(r.query_id, {"query": str(r.query), "q": q, "pos": None, "neg": None})
            if label in POS_LABELS and e["pos"] is None:
                e["pos"] = (str(r.product_id), t, im)
            elif label in NEG_LABELS and e["neg"] is None:
                e["neg"] = (str(r.product_id), t, im)
    examples = []
    skipped = 0
    for qid, e in records.items():
        if e["pos"] is None or e["neg"] is None:
            skipped += 1
            continue
        qa = vectorizer.transform([e["query"]]).tocsr()
        pidp, pt, pi = e["pos"]
        pidn, nt, ni = e["neg"]
        rp, rn = attr_rows.get(pidp), attr_rows.get(pidn)
        pa = float((qa @ attr_matrix[rp].T).toarray()[0, 0]) if rp is not None else 0.0
        na = float((qa @ attr_matrix[rn].T).toarray()[0, 0]) if rn is not None else 0.0
        examples.append((e["q"], pt, pi, pa, nt, ni, na))
    print(f"{name}: query IDs={len(ids):,}; usable pairs={len(examples):,}; skipped={skipped:,}")
    if not examples:
        raise RuntimeError(f"No usable {name} pairs.")
    data = {}
    for j, key in enumerate(["q", "pos_t", "pos_i", "pos_a", "neg_t", "neg_i", "neg_a"]):
        vals = [e[j] for e in examples]
        data[key] = torch.tensor(np.stack(vals) if j in (0,1,2,4,5) else vals, dtype=torch.float32)
    return data

def pairwise_loss(model, d, idx=None):
    if idx is None:
        idx = np.arange(len(d["q"]))
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(idx), 512):
            ix = idx[start:start+512]
            q = d["q"][ix].to(DEVICE)
            w = model(q)
            pos = w[:,0] * (q*d["pos_t"][ix].to(DEVICE)).sum(1) + w[:,1] * (q*d["pos_i"][ix].to(DEVICE)).sum(1) + w[:,2] * d["pos_a"][ix].to(DEVICE)
            neg = w[:,0] * (q*d["neg_t"][ix].to(DEVICE)).sum(1) + w[:,1] * (q*d["neg_i"][ix].to(DEVICE)).sum(1) + w[:,2] * d["neg_a"][ix].to(DEVICE)
            losses.append(float(F.relu(MARGIN - (pos-neg)).mean().item()))
    return float(np.mean(losses))

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)
    print("="*68, "\nM6 THREE-WAY ADAPTIVE ROUTER TRAINING\n", "="*68)
    print("Device:", DEVICE)
    attr = load_npz(ATTR_MATRIX).tocsr()
    ids = np.load(ATTR_IDS, allow_pickle=True)
    vectorizer = joblib.load(VECTORIZER)
    if attr.shape[0] != len(ids):
        raise ValueError("M5 attribute matrix and IDs are not aligned.")
    attr_rows = {str(pid): i for i, pid in enumerate(ids)}
    train_ids = np.load(TRAIN_IDS, allow_pickle=True).tolist()
    val_ids = np.load(VAL_IDS, allow_pickle=True).tolist()
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/validation query ID overlap detected.")
    train = load_pairs(train_ids, "train", vectorizer, attr, attr_rows)
    val = load_pairs(val_ids, "validation", vectorizer, attr, attr_rows)
    model = AdaptiveRouter3().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        order = np.random.permutation(len(train["q"]))
        losses = []
        for start in range(0, len(order), 256):
            ix = order[start:start+256]
            q = train["q"][ix].to(DEVICE)
            w = model(q)
            pos = w[:,0]*(q*train["pos_t"][ix].to(DEVICE)).sum(1) + w[:,1]*(q*train["pos_i"][ix].to(DEVICE)).sum(1) + w[:,2]*train["pos_a"][ix].to(DEVICE)
            neg = w[:,0]*(q*train["neg_t"][ix].to(DEVICE)).sum(1) + w[:,1]*(q*train["neg_i"][ix].to(DEVICE)).sum(1) + w[:,2]*train["neg_a"][ix].to(DEVICE)
            loss = F.relu(MARGIN-(pos-neg)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        tr_loss, val_loss = float(np.mean(losses)), pairwise_loss(model, val)
        history.append({"epoch":epoch,"train_pairwise_loss":tr_loss,"val_pairwise_loss":val_loss})
        print(f"Epoch {epoch:02d}/{EPOCHS} | train={tr_loss:.5f} | val={val_loss:.5f}")
        if val_loss < best - 1e-6:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state_dict":model.state_dict(),"input_dim":DIM,"num_modalities":3,
                        "modalities":["text","image","brand_color_attribute"],"best_epoch":epoch,
                        "best_val_pairwise_loss":best,"seed":SEED,"margin":MARGIN}, CHECKPOINT)
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping.")
                break
    pd.DataFrame(history).to_csv(os.path.join(RESULTS,"m6_training_history.csv"),index=False)
    print(f"\nBest epoch: {best_epoch}; validation loss: {best:.6f}")
    print("Checkpoint:", CHECKPOINT)
    print("CAVEAT: query IDs are internal partitions from a parquet file labelled dataset_split=test.")

if __name__ == "__main__":
    main()
