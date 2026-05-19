# plm_interpretability

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-teal.json)](https://github.com/mila-iqia/ResearchTemplate)

Code for interpretability of protein language models. 

See further documentation in this [Google Doc](https://docs.google.com/document/d/1oX9AFHb4eMkZKJnRwkQR1JHzYjJcCEGedlAJTiG-2e0/edit?usp=sharing)

> [!WARNING]  
> This repo was copied over from a private repo and has not been directly tested. There may be instability.

## Installation

Create a new environment

```bash
uv venv
```

Install the packages

```bash
uv sync
```

## Usage

```console
. .venv/bin/activate
python project/main.py --help
```

## Folders

[algorithms](/algorithms) contains callbacks, lightning modules, and classes for linear probing, including wrapper for protein language models.

[configs] contains config files for running probing experiments using [Hydra](https://hydra.cc/). 

[datamodules](/datamodules) contains dataset classes and dataloaders.

[notebooks](/notebooks) contains Jupyter Notebooks for dataset creation from human proteome information downloaded from UniProt, as well as notebooks for visualizing linear probing, knn, PCA, and interventions results.

[scripts](/scripts) contains folders with the code used for launching linear probing and nearest neighbors experiments with slurm and Hydra, as well as code for caching embeddings from protein language models.

[utils](/utils) contains files to import constants (in strs.py), helper functions, metrics, and splitting strategies.

## Contribution

Install pre-commit hooks

```bash
pre-commit install
```
