"""Dataset, graph-splitting, and Laplacian helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import torch


class NodeSplit(NamedTuple):
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor


DATASETS = {
    "cora": dgl.data.CoraGraphDataset,
    "citeseer": dgl.data.CiteseerGraphDataset,
    "pubmed": dgl.data.PubmedGraphDataset,
}


def load_dataset(name: str, data_dir: str | Path = "data"):
    """Download a supported citation dataset and return it with a bidirected graph."""
    try:
        dataset_class = DATASETS[name]
    except KeyError as error:
        choices = ", ".join(DATASETS)
        raise ValueError(f"unsupported dataset {name!r}; choose from {choices}") from error

    dataset = dataset_class(raw_dir=str(data_dir))
    graph = dgl.to_bidirected(dataset[0], copy_ndata=True)
    return dataset, graph


def create_random_split(num_nodes: int, seed: int) -> NodeSplit:
    """Create the experiment's deterministic 10%/30%/60% node split."""
    if num_nodes < 10:
        raise ValueError("at least 10 nodes are required for a 10/30/60 split")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_nodes, generator=generator)
    train_end = int(num_nodes * 0.1)
    validation_end = int(num_nodes * 0.4)
    return NodeSplit(
        train=indices[:train_end],
        validation=indices[train_end:validation_end],
        test=indices[validation_end:],
    )


def build_left_normalized_laplacian(graph) -> torch.Tensor:
    """Construct L = I - D^-1 A as a coalesced Torch sparse tensor."""
    sources, destinations = graph.edges()
    device = sources.device
    values = torch.ones(sources.shape[0], device=device, dtype=torch.float32)

    row_sums = torch.zeros(graph.num_nodes(), device=device, dtype=torch.float32)
    row_sums.scatter_add_(0, sources, values)
    inverse_degrees = torch.zeros_like(row_sums)
    nonzero = row_sums > 0
    inverse_degrees[nonzero] = row_sums[nonzero].reciprocal()

    diagonal = torch.arange(graph.num_nodes(), device=device)
    indices = torch.cat(
        (
            torch.stack((diagonal, diagonal)),
            torch.stack((sources, destinations)),
        ),
        dim=1,
    )
    laplacian_values = torch.cat(
        (torch.ones_like(row_sums), -inverse_degrees[sources])
    )
    return torch.sparse_coo_tensor(
        indices,
        laplacian_values,
        (graph.num_nodes(), graph.num_nodes()),
        device=device,
    ).coalesce()