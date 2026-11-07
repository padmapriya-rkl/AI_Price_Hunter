\# AI Price Hunter



AI Price Hunter is an intelligent product retrieval and price comparison system that combines multimodal search techniques with cross-platform product matching.



The project explores text-based retrieval, image-based retrieval, multimodal fusion, adaptive routing, and product matching across Amazon and Walmart datasets.



\## Project Overview



The system consists of two major components:



1\. \*\*Multimodal Product Retrieval:\*\* Retrieves relevant products using text, image, and product attributes.

2\. \*\*Cross-Platform Product Matching:\*\* Identifies matching products across Amazon and Walmart and compares their recorded prices.



\## Retrieval Models



| Model | Description |

|---|---|

| M1 | Text-based retrieval using Sentence Transformers |

| M2 | CLIP-based image retrieval |

| M3 | Static text-image fusion |

| M4 | Adaptive text-image fusion |

| M5 | Static three-way fusion using text, image, and attributes |

| M6 | Adaptive three-way fusion using a query-conditioned router |



M6 uses a neural router to determine the contribution of text, image, and attribute representations.



\## Cross-Platform Product Matching



The Amazon–Walmart component uses product information such as:



\- Product titles and descriptions

\- Brand and model information

\- Category and product attributes

\- TF-IDF similarity features

\- Pairwise classification



The matching component identifies candidate product pairs and compares their recorded prices.



\*Note: Price comparisons depend on the available dataset records and do not necessarily represent live marketplace prices.\*



\## Repository Structure



```text

AI\_Price\_Hunter/

│

├── src/                 # Python source code

│

├── data/                # Datasets, embeddings, indexes,

│                        # checkpoints and generated data

│

├── results/             # Model evaluation outputs

│

├── README.md

├── requirements.txt

└── .gitignore

```



\## Datasets



The project uses:



1\. \*\*SQID-enriched Amazon ESCI dataset\*\* – Used for product retrieval experiments and relevance evaluation.

2\. \*\*Amazon–Walmart product matching dataset\*\* – Used for cross-platform product matching and price comparison.



The ESCI relevance labels include Exact, Substitute, Complement, and Irrelevant.



Large datasets and restricted or licensed files may not be included in this repository. Obtain the required datasets separately and place them in the paths expected by the corresponding scripts.



\## Experimental Results



The repository's `results/` directory contains evaluation summaries and per-query outputs for the retrieval experiments.



The evaluated metrics include:



\- HitRate@10

\- Recall@10

\- nDCG@10



Refer to the individual result files and project report for the corresponding experimental configurations and metric definitions.



\## M6 Experimental Notes



\- M6 uses a three-output query-conditioned router.

\- The router combines text, image, and brand-colour attribute information.

\- Training uses pairwise comparisons based on Exact/Substitute and Irrelevant/Complement relevance labels.

\- The query partitions used in the experiment are internal random partitions, not official independent dataset splits.

\- Attribute similarity can be zero when a product is absent from the attribute index.

\- Training and evaluation use different score-scaling procedures; interpret the results accordingly.



\## Requirements



Install the dependencies listed in `requirements.txt`:



```bash

pip install -r requirements.txt

```



Some experiments may require additional datasets, trained checkpoints, embeddings, FAISS indexes, and attribute files.



\## Running the Project



Run scripts from the project root so that project-relative paths can be resolved:



```bash

python .\\src\\search\_products.py

```



For model training and evaluation, use the corresponding scripts in `src/` after placing all required datasets and artifacts in their expected locations.



\## Technologies



\- Python

\- PyTorch

\- Sentence Transformers

\- CLIP

\- FAISS

\- Scikit-learn

\- Pandas

\- NumPy



\## Notes



This repository contains source code and selected experimental artifacts. Dataset availability, model checkpoints, and index files may be subject to separate storage or access requirements.

