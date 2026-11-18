"""Датасет модульного сложения: все пары (x1, x2), one-hot представление, фиксированное разбиение."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import is_prime


@dataclass(frozen=True)
class ModularAdditionData:
    q: int
    X_train: torch.Tensor  # (N_train, 2q), float32, one-hot(x1) ⊕ one-hot(x2)
    Y_train: torch.Tensor  # (N_train, q), float32, one-hot(y)
    X_test: torch.Tensor
    Y_test: torch.Tensor
    pairs_train: torch.Tensor  # (N_train, 2), int64 — исходные (x1, x2) для анализа
    pairs_test: torch.Tensor


def make_modular_addition(q: int, train_frac: float, split_seed: int) -> ModularAdditionData:
    """Строит все q*q пар, y = (x1 + x2) mod q, и делит их на train/test.

    Разбиение зависит только от split_seed: одинаково для всех реплик и экспериментов,
    поэтому разброс между seed отражает случайность инициализации, а не выборки.
    """
    if not is_prime(q):
        raise ValueError(f"q = {q} должно быть простым")
    if not 0 < train_frac < 1:
        raise ValueError("train_frac должна быть в (0, 1)")

    x1, x2 = torch.meshgrid(torch.arange(q), torch.arange(q), indexing="ij")
    x1, x2 = x1.reshape(-1), x2.reshape(-1)
    y = (x1 + x2) % q
    X = torch.cat([F.one_hot(x1, q), F.one_hot(x2, q)], dim=1).float()
    Y = F.one_hot(y, q).float()
    pairs = torch.stack([x1, x2], dim=1)

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(q * q, generator=g)
    n_train = int(round(train_frac * q * q))
    idx_train, idx_test = perm[:n_train], perm[n_train:]
    return ModularAdditionData(
        q=q,
        X_train=X[idx_train],
        Y_train=Y[idx_train],
        X_test=X[idx_test],
        Y_test=Y[idx_test],
        pairs_train=pairs[idx_train],
        pairs_test=pairs[idx_test],
    )
