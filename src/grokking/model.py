"""BatchedMLP: R независимых полносвязных сетей, веса которых хранятся в общих тензорах."""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch


class BatchedMLP(torch.nn.Module):
    """R независимых MLP вида d_in -> H -> ... -> H -> d_out с ReLU.

    Веса слоя l хранятся тензором (R, d_in_l, d_out_l), смещения — (R, d_out_l).
    Реплики не влияют друг на друга при обучении, потому что:
      * общий лосс — сумма лоссов реплик, значит градиент по весам реплики r
        зависит только от её собственного лосса;
      * Adam обновляет каждый элемент тензора независимо;
      * глобальный клиппинг градиента не используется (он связал бы реплики).

    Инициализация каждой реплики — как у nn.Linear (равномерно в ±1/sqrt(fan_in)),
    но со своим генератором: веса реплики зависят только от её seed.
    """

    def __init__(self, seeds: Sequence[int], d_in: int, d_hidden: int, d_out: int, n_hidden_layers: int = 1):
        super().__init__()
        n_replicas = len(seeds)
        dims = [d_in] + [d_hidden] * n_hidden_layers + [d_out]
        weights = [torch.empty(n_replicas, a, b) for a, b in itertools.pairwise(dims)]
        biases = [torch.empty(n_replicas, b) for b in dims[1:]]

        for r, seed in enumerate(seeds):
            g = torch.Generator().manual_seed(int(seed))
            for W, b, fan_in in zip(weights, biases, dims[:-1]):
                bound = 1.0 / math.sqrt(fan_in)
                W[r].uniform_(-bound, bound, generator=g)
                b[r].uniform_(-bound, bound, generator=g)

        self.weights = torch.nn.ParameterList([torch.nn.Parameter(W) for W in weights])
        self.biases = torch.nn.ParameterList([torch.nn.Parameter(b) for b in biases])

    @property
    def n_replicas(self) -> int:
        return self.weights[0].shape[0]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, d_in), общий для всех реплик. Возвращает логиты (R, N, d_out).

        Первый слой — умножение one-hot входа на W1, как в постановке задачи.
        """
        h = X.unsqueeze(0).expand(self.n_replicas, -1, -1)
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b.unsqueeze(1), h, W)
            if i < n_layers - 1:
                h = torch.relu(h)
        return h

    def weight_matrices(self) -> list[torch.Tensor]:
        """Матрицы весов всех слоёв, формы (R, d_in_l, d_out_l); смещения не входят."""
        return list(self.weights)
