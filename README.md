# PottsNet: Node Classification Example

This folder is a minimal, runnable release of PottsNet for transductive node
classification. It includes the parallel and sequential PottsNet updates and a
small GCN baseline. The example supports Cora, Citeseer, and Pubmed through DGL.

## Models

- `potts-par`: applies graph convolution, region force, and graph diffusion in
  one parallel update.
- `potts-seq`: applies graph convolution first, followed by region force and
  graph diffusion.
- `gcn`: a one-layer GCN encoder with the same graph-convolution classifier.

Both PottsNet variants use two update steps, a one-layer GCN region-force
network, and a GraphSAGE mean operator for the learned update.

## Installation

Python 3.9 through 3.11 is supported. Create and activate a virtual environment,
then install a matching PyTorch and DGL pair. DGL wheels are platform-specific;
follow the [DGL installation guide](https://www.dgl.ai/pages/start.html) when a
CUDA build is required.

For a CPU environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux or macOS
source .venv/bin/activate

pip install -r requirements.txt
```

If the default DGL package does not support your PyTorch version, install the
compatible versions first and then run `pip install numpy pytest`.

## Quick Start

Run commands from this folder. Dataset files are downloaded automatically to
`data/` and are ignored by Git.

```bash
# Parallel PottsNet on CPU
python train_node_classification.py --model potts-par --dataset cora --gpu -1

# Sequential PottsNet on GPU 0
python train_node_classification.py --model potts-seq --dataset citeseer --gpu 0

# Short GCN smoke run
python train_node_classification.py --model gcn --epochs 1 --runs 1 --gpu -1
```

Use `python train_node_classification.py --help` for all options. The default
experiment runs five seeds and reports the mean and population standard
deviation of test accuracy.

## Evaluation Protocol

For each run, the nodes are randomly divided into 10% training, 30% validation,
and 60% test sets. The split and feature noise are reproducible from `--seed`;
run `i` uses seed `seed + i`. This is a custom experimental split, not the
canonical citation-dataset masks.

When `--node-noise` is positive, Gaussian noise is added once to the node
features for each run. Features are then clipped to non-negative values and
row-normalized. The best epoch is selected using validation accuracy only. Its
weights are restored before the test split is evaluated once.

## Tests

The tests use a synthetic graph and do not download data:

```bash
pytest -q
```

They check deterministic splits, sparse Laplacian construction, and one full
forward/backward update for every model.

## Layout

```text
pottsnet_example/
|-- data.py                       # Dataset, split, and Laplacian helpers
|-- models.py                     # PottsNet and GCN modules
|-- train_node_classification.py  # Training CLI
|-- tests/test_smoke.py           # Offline smoke tests
|-- requirements.txt
`-- LICENSE
```

## Citation

Add the associated paper citation here before publishing the repository.

## License

This example is released under the MIT License. See `LICENSE`.