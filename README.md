# KnowRA
These are the source codes and datasets for IJCAI submission 255, ***KnowRA: Knowledge Retrieval Augmented Method for Document-level Relation Extraction with Comprehensive Reasoning Abilities***, for IJCAI 2025.

## Requirements

- Python     (tested on 3.10.9)
- CUDA     (tested on 11.8)
- PyTorch     (tested on 2.0.0)
- Transformers     (tested on 4.20.1)
- numpy     (tested on 1.23.5)
- apex     (tested on 0.1)
- opt-einsum     (tested on 3.3.0)
- hydra-core     (tested on 1.3.2)
- ujson
- spacy
- tqdm

## Description

- ``configs/``: Code file for model parameter configuration.
- ``data/``: Folder for input datasets.
  - `gen_coref.py`: Code file for generate co-reference reasoning documents.
  - `gen_graph.py`: Code file for generate knowledge enhanced documents.

- ``datasets.py``: Code file for data processing and evaluation metrics.
- ``long_seq.py``: Code file for input context length processing for model.
- ``model.py``: Code file for various baseline relation extraction models.
- ``train.py``, ``train_docred.py``: Main code file for model training.
- ``utils.py``: Code file for co-reference reasoning and knowledge graph processing.

## Datasets

We experiment our model on two public datasets: **Re-DocRED** and **DWIE**.

For the both datasets,  you can generate the co-reference reasoning and knowledge enhanced documents with`gen_coref.py` and `gen_graph.py`.

## Training

1. Change you configurations  with the `.yaml` profile in the `configs/` folder.
2. Train KnowRA model with the following comand: `python train.py`
