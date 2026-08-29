"""Train PottsNet or a GCN baseline on a citation network."""

from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from data import DATASETS, build_left_normalized_laplacian, create_random_split, load_dataset
from models import build_model


MODEL_NAMES = ("potts-par", "potts-seq", "gcn")


@dataclass(frozen=True)
class RunResult:
    seed: int
    best_epoch: int
    validation_accuracy: float
    test_accuracy: float
    parameters: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transductive node classification with PottsNet"
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="cora")
    parser.add_argument("--model", choices=MODEL_NAMES, default="gcn")
    parser.add_argument("--eta", type=float, default=1.0, help="diffusion strength")
    parser.add_argument("--node-noise", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="seed for the first run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; use -1 for CPU")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)

    if args.epochs < 1 or args.runs < 1:
        parser.error("--epochs and --runs must be positive")
    if args.node_noise < 0:
        parser.error("--node-noise must be non-negative")
    return args


def resolve_device(gpu_index: int) -> torch.device:
    if gpu_index < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_index >= torch.cuda.device_count():
        print(f"GPU {gpu_index} is unavailable; using CPU.")
        return torch.device("cpu")

    device = torch.device(f"cuda:{gpu_index}")
    try:
        torch.zeros(1, device=device)
        dgl.graph(([0], [0])).to(device)
    except Exception as error:
        print(f"CUDA is incompatible with the installed Torch/DGL stack; using CPU: {error}")
        return torch.device("cpu")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_feature_noise(features: torch.Tensor, standard_deviation: float) -> torch.Tensor:
    if standard_deviation == 0:
        return features.clone()

    noisy_features = torch.relu(
        features + torch.randn_like(features) * standard_deviation
    )
    row_sums = noisy_features.sum(dim=1, keepdim=True)
    return torch.where(
        row_sums > 0,
        noisy_features / row_sums.clamp_min(torch.finfo(features.dtype).eps),
        noisy_features,
    )


def accuracy(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> float:
    predictions = logits[indices].argmax(dim=1)
    return (predictions == labels[indices]).float().mean().item()


def train_one_run(args: argparse.Namespace, dataset, source_graph, seed: int, device: torch.device) -> RunResult:
    seed_everything(seed)
    graph = source_graph.to(device)
    graph.laplacian = build_left_normalized_laplacian(graph)
    split = create_random_split(graph.num_nodes(), seed)
    split = split._replace(
        train=split.train.to(device),
        validation=split.validation.to(device),
        test=split.test.to(device),
    )

    features = add_feature_noise(graph.ndata["feat"].float(), args.node_noise)
    labels = graph.ndata["label"].long()
    model = build_model(
        args.model,
        input_dim=features.shape[1],
        num_classes=dataset.num_classes,
        eta=args.eta,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[split.train], labels[split.train])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(graph, features)
            validation_accuracy = accuracy(
                validation_logits, labels, split.validation
            )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"  epoch {epoch:03d} | loss {loss.item():.4f} | "
                f"validation {validation_accuracy:.4f} | "
                f"best {best_validation_accuracy:.4f}"
            )

    if best_state is None:
        raise RuntimeError("training completed without a valid checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(graph, features)
        test_accuracy = accuracy(test_logits, labels, split.test)

    return RunResult(
        seed=seed,
        best_epoch=best_epoch,
        validation_accuracy=best_validation_accuracy,
        test_accuracy=test_accuracy,
        parameters=parameter_count,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.gpu)
    dataset, graph = load_dataset(args.dataset, args.data_dir)
    print(
        f"Dataset: {args.dataset} | model: {args.model} | device: {device} | "
        f"split: 10/30/60 | runs: {args.runs}"
    )

    results = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        print(f"Run {run_index + 1}/{args.runs} (seed={seed})")
        result = train_one_run(args, dataset, graph, seed, device)
        results.append(result)
        print(
            f"  selected epoch {result.best_epoch} | "
            f"validation {result.validation_accuracy:.4f} | "
            f"test {result.test_accuracy:.4f}"
        )

    test_accuracies = np.asarray([result.test_accuracy for result in results])
    print(
        f"Test accuracy: {test_accuracies.mean():.4f} +/- "
        f"{test_accuracies.std():.4f} | parameters: {results[0].parameters}"
    )


if __name__ == "__main__":
    main()