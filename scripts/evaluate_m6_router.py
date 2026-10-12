import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
from scipy.sparse import load_npz
import joblib
import torch
import torch.nn as nn

BASE = r"D:\DLPROJECT"
DATA = os.path.join(BASE,"data","processed","multimodal")
M4, M5 = os.path.join(DATA,"m4"), os.path.join(DATA,"m5")
PARQUET = os.path.join(DATA,"sqid_esci_test.parquet")
TEST_IDS = os.path.join(M4,"test_query_ids.npy")
CHECKPOINT = os.path.join(M5,"m6_adaptive_router.pt")
TEXT_INDEX = os.path.join(BASE,"data","indexes","m4_clip_text_faiss.index")
TEXT_IDS = os.path.join(DATA,"m4_clip_text_product_ids.npy")
IMAGE_INDEX = os.path.join(BASE,"data","indexes","clip_image_faiss.index")
IMAGE_IDS = os.path.join(DATA,"clip_image_product_ids.npy")
ATTR_MATRIX = os.path.join(M5,"attribute_tfidf.npz")
ATTR_IDS = os.path.join(M5,"attribute_product_ids.npy")
VECTORIZER = os.path.join(M5,"attribute_vectorizer.pkl")
RESULTS = os.path.join(BASE,"results")
SUMMARY = os.path.join(RESULTS,"m6_adaptive_three_way_results.csv")
PER_QUERY = os.path.join(RESULTS,"m6_adaptive_three_way_per_query.csv")
DIM, BATCH_SIZE, TOP_CANDIDATES, TOP_K = 768, 5000, 200, 10
RELEVANT = {"E","S"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdaptiveRouter3(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(DIM,128),nn.ReLU(),nn.Linear(128,3))
    def forward(self,x):
        return torch.softmax(self.network(x),dim=1)

def vec(x):
    if x is None: return None
    try: x=np.asarray(x,dtype=np.float32)
    except (TypeError,ValueError): return None
    if x.shape!=(DIM,) or not np.isfinite(x).all(): return None
    n=float(np.linalg.norm(x))
    return None if n<1e-12 else x/n

def norm_scores(d):
    if not d: return {}
    v=np.asarray(list(d.values()),dtype=np.float32); lo,hi=float(v.min()),float(v.max())
    if hi-lo<1e-8:
        x=float(np.clip(v[0],0,1)); return {k:x for k in d}
    return {k:float((x-lo)/(hi-lo)) for k,x in d.items()}

def main():
    os.makedirs(RESULTS,exist_ok=True)
    print("="*68,"\nM6 ADAPTIVE THREE-WAY EVALUATION\n","="*68)
    print("Device:",DEVICE)
    model=AdaptiveRouter3().to(DEVICE)
    ckpt=torch.load(CHECKPOINT,map_location=DEVICE,weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt,dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()
    print("Loaded checkpoint; best epoch:",ckpt.get("best_epoch","unknown") if isinstance(ckpt,dict) else "unknown")
    ti=faiss.read_index(TEXT_INDEX); ii=faiss.read_index(IMAGE_INDEX)
    tids=np.load(TEXT_IDS,allow_pickle=True); iids=np.load(IMAGE_IDS,allow_pickle=True)
    if len(tids)!=ti.ntotal or len(iids)!=ii.ntotal: raise ValueError("FAISS index/ID mapping mismatch.")
    am=load_npz(ATTR_MATRIX).tocsr(); aids=np.load(ATTR_IDS,allow_pickle=True); vectorizer=joblib.load(VECTORIZER)
    if am.shape[0]!=len(aids): raise ValueError("Attribute matrix/ID mismatch.")
    arows={str(pid):i for i,pid in enumerate(aids)}
    test=set(np.load(TEST_IDS,allow_pickle=True).tolist())
    queries,gt={},{}
    cols=["query_id","query","product_id","product_locale","esci_label","sqid_query_clip_text_features"]
    for batch in pq.ParquetFile(PARQUET).iter_batches(batch_size=BATCH_SIZE,columns=cols):
        df=batch.to_pandas(); df=df[df.query_id.isin(test)]
        for r in df.itertuples(index=False):
            if r.query_id not in queries:
                q=vec(r.sqid_query_clip_text_features)
                if q is not None: queries[r.query_id]={"query":str(r.query),"q":q}
            if str(r.esci_label).strip() in RELEVANT and str(r.product_locale).lower().strip()=="us":
                gt.setdefault(r.query_id,set()).add(f"{r.product_id}|||us")
    qids=[qid for qid in queries if gt.get(qid)]
    if not qids: raise RuntimeError("No evaluable test queries found.")
    print(f"Evaluable test queries: {len(qids):,}")
    out=[]
    with torch.no_grad():
        for n,qid in enumerate(qids,1):
            d=queries[qid]; q=d["q"]; q2=q.reshape(1,DIM).astype(np.float32)
            w=model(torch.from_numpy(q).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
            ts,tx=ti.search(q2,TOP_CANDIDATES); ims,ix=ii.search(q2,TOP_CANDIDATES)
            tr={}
            for s,j in zip(ts[0],tx[0]):
                if j>=0:
                    key=str(tids[j])
                    if "|||" in key:
                        p,l=key.split("|||",1); key=f"{p}|||{l.lower()}"
                    tr[key]=float(s)
            ir={f"{iids[j]}|||us":float(s) for s,j in zip(ims[0],ix[0]) if j>=0}
            tn,inn=norm_scores(tr),norm_scores(ir)
            candidates=set(tr)|set(ir); qa=vectorizer.transform([d["query"]]).tocsr()
            ar={}
            for key in candidates:
                row=arows.get(key.split("|||",1)[0])
                if row is not None: ar[key]=float((qa@am[row].T).toarray()[0,0])
            ranked=[]
            for key in candidates:
                st,si,sa=tn.get(key,0.0),inn.get(key,0.0),ar.get(key,0.0)
                score=float(w[0])*st+float(w[1])*si+float(w[2])*sa
                ranked.append((key,score,st,si,sa))
            ranked.sort(key=lambda x:x[1],reverse=True); top=ranked[:TOP_K]
            ids=[x[0] for x in top]; relevant=gt[qid]; hits=sum(x in relevant for x in ids)
            hit=float(hits>0); recall=hits/len(relevant)
            dcg=sum(1/np.log2(rank+1) for rank,p in enumerate(ids,1) if p in relevant)
            ideal=min(len(relevant),TOP_K); idcg=sum(1/np.log2(rank+1) for rank in range(1,ideal+1))
            ndcg=dcg/idcg if idcg else 0.0
            best=top[0] if top else (None,0,0,0,0)
            out.append({"query_id":qid,"query":d["query"],"candidate_count":len(candidates),
                "relevant_product_count":len(relevant),"top10_hits":hits,"hit_rate_at_10":hit,
                "recall_at_10":recall,"ndcg_at_10":ndcg,"weight_text":float(w[0]),
                "weight_image":float(w[1]),"weight_attribute":float(w[2]),
                "top1_product_id_locale":best[0],"top1_fused_score":best[1],
                "top1_text_score":best[2],"top1_image_score":best[3],"top1_attribute_score":best[4]})
            if n%100==0 or n==len(qids): print(f"Processed {n:,}/{len(qids):,}")
    df=pd.DataFrame(out); df.to_csv(PER_QUERY,index=False)
    pd.DataFrame([{"method":"M6 Adaptive Text+Image+BrandColor","queries_evaluated":len(df),
        "text_products_indexed":ti.ntotal,"image_products_indexed":ii.ntotal,"attribute_products_indexed":len(aids),
        "top_candidates_per_modality":TOP_CANDIDATES,"top_k":TOP_K,
        "hit_rate_at_10":df.hit_rate_at_10.mean(),"recall_at_10":df.recall_at_10.mean(),"ndcg_at_10":df.ndcg_at_10.mean(),
        "mean_weight_text":df.weight_text.mean(),"mean_weight_image":df.weight_image.mean(),
        "mean_weight_attribute":df.weight_attribute.mean(),
        "split_note":"Internal query partition from parquet labelled dataset_split=test"}]).to_csv(SUMMARY,index=False)
    print("\nM6 RESULTS")
    print(f"HitRate@10: {df.hit_rate_at_10.mean():.4f}")
    print(f"Recall@10:  {df.recall_at_10.mean():.4f}")
    print(f"nDCG@10:    {df.ndcg_at_10.mean():.4f}")
    print("Mean weights:",df[["weight_text","weight_image","weight_attribute"]].mean().to_dict())
    print("Summary:",SUMMARY); print("Per-query:",PER_QUERY)
    print("For fair comparison, rerun M1-M5 using the same test IDs, candidate protocol, score normalization, and metrics.")
    print("CAVEAT: these are internal partitions, not official dataset splits.")

if __name__=="__main__":
    main()
