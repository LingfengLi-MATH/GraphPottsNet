"""Neural network modules used by the node-classification example."""

import os

os.environ.setdefault("DGLBACKEND", "pytorch")

import torch
from dgl.nn import GraphConv, SAGEConv
from torch import nn


class GCNEncoder(nn.Module):
    """A small GCN encoder used as a baseline and region-force network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        dimensions = [input_dim] + [hidden_dim] * depth
        self.layers = nn.ModuleList(
            GraphConv(dimensions[index], dimensions[index + 1])
            for index in range(depth)
        )
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(hidden_dim) for _ in range(depth - 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = torch.tanh(layer(graph, hidden))
            if index < len(self.batch_norms):
                hidden = self.dropout(self.batch_norms[index](hidden))
        return hidden


class _PottsNetBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int = 2,
        eta: float = 5.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.embedding = GraphConv(input_dim, hidden_dim)
        self.region_force = GCNEncoder(
            input_dim, hidden_dim, depth=1, dropout=0.0
        )
        self.update_layers = nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim, "mean") for _ in range(depth)
        )
        self.time_step = 1.0 / depth
        self.eta = eta

    @staticmethod
    def _laplacian_product(graph, features: torch.Tensor) -> torch.Tensor:
        if not hasattr(graph, "laplacian"):
            raise AttributeError(
                "graph.laplacian is required; call build_left_normalized_laplacian first"
            )
        return torch.sparse.mm(graph.laplacian, features)


class PottsNetParallel(_PottsNetBase):
    """PottsNet with parallel graph-convolution and diffusion updates."""

    def forward(self, graph, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.embedding(graph, features))
        region_force = torch.tanh(self.region_force(graph, features))

        for update_layer in self.update_layers:
            convolution = torch.relu(update_layer(graph, hidden))
            hidden = torch.tanh(
                hidden
                - self.time_step * convolution
                - self.time_step * region_force
                - self.eta
                * self.time_step
                * self._laplacian_product(graph, hidden)
            )
        return hidden


class PottsNetSequential(_PottsNetBase):
    """PottsNet with a sequential convolution-then-diffusion update."""

    def forward(self, graph, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.embedding(graph, features))
        region_force = torch.tanh(self.region_force(graph, features))

        for update_layer in self.update_layers:
            convolution = torch.relu(update_layer(graph, hidden))
            intermediate = torch.tanh(hidden + convolution)
            hidden = torch.tanh(
                intermediate
                - self.time_step * region_force
                - self.eta
                * self.time_step
                * self._laplacian_product(graph, intermediate)
            )
        return hidden


class NodeClassifier(nn.Module):
    """Combine a node encoder with a graph-convolution classification head."""

    def __init__(self, encoder: nn.Module, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.output_layer = GraphConv(hidden_dim, num_classes)

    def forward(self, graph, features: torch.Tensor) -> torch.Tensor:
        return self.output_layer(graph, self.encoder(graph, features))


def build_model(
    name: str,
    input_dim: int,
    num_classes: int,
    eta: float = 5.0,
) -> NodeClassifier:
    """Build one of the models supported by the public example."""
    if name == "gcn":
        hidden_dim = 32
        encoder = GCNEncoder(input_dim, hidden_dim, depth=1)
    elif name == "potts-par":
        hidden_dim = 16
        encoder = PottsNetParallel(input_dim, hidden_dim, depth=2, eta=eta)
    elif name == "potts-seq":
        hidden_dim = 16
        encoder = PottsNetSequential(input_dim, hidden_dim, depth=2, eta=eta)
    else:
        raise ValueError(f"unsupported model: {name}")

    return NodeClassifier(encoder, hidden_dim, num_classes)