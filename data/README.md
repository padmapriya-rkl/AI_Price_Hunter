I Price Hunter processes approximately 1.8 million product records to support product retrieval, similarity matching, price comparison, and multimodal search experiments.

This directory contains the datasets, embeddings, indexes, metadata, and intermediate artifacts used throughout the project.

Data Organization

The local data directory is organized into the following components:

Directory / File

Description

amazon_walmart_*

Amazon and Walmart product data and related processing artifacts

embeddings/

Product text embeddings and associated metadata

embeddings_us_full_v1/

Full-scale product embedding artifacts

esci/

ESCI dataset files used for product search and relevance experiments

multimodal/

Multimodal features, image embeddings, and fusion-related artifacts

splits/

Dataset train/test split information

text_baseline/

Text-based retrieval experiment data

*.index

FAISS indexes used for similarity search

*.npy

NumPy arrays containing embeddings, IDs, and numerical features

*.parquet

Structured product and query datasets

Note: The complete dataset and generated artifacts are maintained locally and are not all included in this GitHub repository because of their size.

Dataset Scale

Approximately 1.8 million product records are processed.

Multiple product and query datasets are used for experimentation.

Embedding arrays and FAISS indexes support efficient similarity-based retrieval.

Both text-based and multimodal retrieval components are evaluated.

Why Full Data Is Not Included

The complete data directory occupies approximately 12.4 GB. Large datasets, embeddings, and generated indexes are excluded from the repository to avoid excessive repository storage and Git LFS usage.

The source code and experiment results are maintained in GitHub. The complete data artifacts are retained locally.
