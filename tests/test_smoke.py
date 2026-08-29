"""Offline smoke tests for the public node-classification example."""

import os
import sys
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgl
import pytest
import torch
import torch.nn.functional as F

from data import build_left_normalized_laplacian, create_random_split
from models import build_model


def make_graph():
    sources = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0])
    destinations = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5])
    graph = dgl.graph((sources, destinations), num_nodes=10)
    graph = dgl.add_self_loop(graph)
    graph.laplacian = build_left_normalized_laplacian(graph)
    return graph


def test_laplacian_is_finite_sparse_matrix():
    graph = make_graph()
    laplacian = graph.laplacian

    assert laplacian.shape == (graph.num_nodes(), graph.num_nodes())
    assert laplacian.is_sparse
    assert laplacian.is_coalesced()
    assert torch.isfinite(laplacian.values()).all()


def test_random_split_is_deterministic_and_disjoint():
    first = create_random_split(100, seed=7)
    second = create_random_split(100, seed=7)

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert (len(first.train), len(first.validation), len(first.test)) == (10, 30, 60)
    combined = torch.cat(first)
    assert combined.unique().numel() == 100


@pytest.mark.parametrize("model_name", ["gcn", "potts-par", "potts-seq"])
def test_model_completes_training_step(model_name):
    torch.manual_seed(0)
    graph = make_graph()
    features = torch.randn(graph.num_nodes(), 4)
    labels = torch.arange(graph.num_nodes()) % 3
    model = build_model(model_name, input_dim=4, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    logits = model(graph, features)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert logits.shape == (graph.num_nodes(), 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)