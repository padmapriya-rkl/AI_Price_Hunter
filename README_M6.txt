# M6 Adaptive Three-Way Fusion

## Files
- `train_m6_router.py`: trains a 3-output query-conditioned router and chooses a checkpoint by validation pairwise loss.
- `evaluate_m6_router.py`: evaluates the saved router using text + image + brand/color attributes.

## Required existing files
- `data/processed/multimodal/m4/train_query_ids.npy`
- `data/processed/multimodal/m4/val_query_ids.npy`
- `data/processed/multimodal/m4/test_query_ids.npy`
- `data/processed/multimodal/sqid_esci_test.parquet`
- `data/processed/multimodal/m5/attribute_tfidf.npz`
- `data/processed/multimodal/m5/attribute_product_ids.npy`
- `data/processed/multimodal/m5/attribute_vectorizer.pkl`
- `data/processed/multimodal/m4_clip_text_product_ids.npy`
- `data/processed/multimodal/clip_image_product_ids.npy`
- `data/indexes/m4_clip_text_faiss.index`
- `data/indexes/clip_image_faiss.index`

The scripts assume the existing M5 filenames and paths match the user's project. Verify these files exist before running.

## Run in PowerShell
```powershell
python .\scripts\train_m6_router.py
python .\scripts\evaluate_m6_router.py
```

Copy both `.py` files into `D:\DLPROJECT\scripts` first.

## Important experimental notes
1. The query splits are internal random partitions from `sqid_esci_test.parquet`, whose `dataset_split` is labelled `test`; they are not official training/validation/test splits.
2. Training chooses the first valid E/S positive and I/C negative per query. This is a simple pairwise training setup, not exhaustive pair mining.
3. Attributes are only brand and color. Attribute similarity may be zero when a product is missing from the M5 attribute index.
4. Evaluation min-max normalizes text and image scores per query, while attribute TF-IDF similarity is cosine-like. Training uses raw normalized embedding similarities plus attribute similarity. Interpret the training objective cautiously; a future refinement should align training-time score scaling with evaluation-time score scaling.
5. M6 metrics alone do not prove improvement. Compare M1-M5 using identical query IDs, candidate generation, score normalization, relevant labels, and metric definitions.
