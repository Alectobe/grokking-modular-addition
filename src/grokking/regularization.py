"""Штрафы L1 и L2 по репликам и показатели весов, которые пишутся в метрики."""
from __future__ import annotations

import torch

from .model import BatchedMLP

SMALL_WEIGHT = 1e-3  # порог «почти нулевого» веса для доли frac_small


def weight_norms(model: BatchedMLP) -> tuple[torch.Tensor, torch.Tensor]:
    """Σ|w| и Σw² по всем матрицам весов (без смещений) для каждой реплики. Формы (R,)."""
    l1 = sum(W.abs().sum(dim=(1, 2)) for W in model.weight_matrices())
    l2sq = sum((W**2).sum(dim=(1, 2)) for W in model.weight_matrices())
    return l1, l2sq


def penalty(model: BatchedMLP, kind: str) -> torch.Tensor:
    """R(θ) для каждой реплики, форма (R,). L1: Σ|w|, L2: Σw², none: нули.

    В лосс штраф входит как ce + rho * penalty, где rho — вектор (R,).
    """
    if kind == "l1":
        return sum(W.abs().sum(dim=(1, 2)) for W in model.weight_matrices())
    if kind == "l2":
        return sum((W**2).sum(dim=(1, 2)) for W in model.weight_matrices())
    if kind == "none":
        W = model.weight_matrices()[0]
        return torch.zeros(model.n_replicas, dtype=W.dtype, device=W.device)
    raise ValueError(f"неизвестный тип регуляризации: {kind}")


def frac_small(model: BatchedMLP, threshold: float = SMALL_WEIGHT) -> torch.Tensor:
    """Доля весов с |w| < threshold для каждой реплики, форма (R,)."""
    n_small = sum((W.abs() < threshold).sum(dim=(1, 2)) for W in model.weight_matrices())
    n_total = sum(W[0].numel() for W in model.weight_matrices())
    return n_small.float() / n_total
