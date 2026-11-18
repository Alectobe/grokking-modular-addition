"""Цикл обучения одного задания: R реплик, полнобатчевый Adam, запись метрик."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import storage
from .config import Config
from .data import make_modular_addition
from .model import BatchedMLP
from .regularization import frac_small, penalty, weight_norms
from .utils import enable_determinism, environment_info, resolve_device, set_global_seed

# Ключи metrics.npz по контракту из docs/ARCHITECTURE.md §5 (кроме rho и seed)
METRIC_KEYS = (
    "step", "train_loss", "test_loss", "train_acc", "test_acc", "penalty", "l1_norm", "l2_norm", "frac_small",
)


def cross_entropy_per_replica(logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Перекрёстная энтропия с one-hot целью, усреднённая по объектам. Форма (R,)."""
    return -(Y.unsqueeze(0) * torch.log_softmax(logits, dim=-1)).sum(-1).mean(-1)


def accuracy_per_replica(logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Доля объектов, у которых argmax логитов совпал с классом. Форма (R,)."""
    return (logits.argmax(-1) == Y.argmax(-1).unsqueeze(0)).float().mean(-1)


def make_optimizer(model: BatchedMLP, cfg: Config) -> torch.optim.Optimizer:
    """Adam или SGD без встроенного weight decay: регуляризация — только явным штрафом."""
    if cfg.optimizer.name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)
    return torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr)


def train_job(cfg: Config, out_dir: Path, log_every_fraction: float = 0.1) -> dict[str, np.ndarray]:
    """Обучает все реплики задания и сохраняет результаты в out_dir.

    Обучение полнобатчевое: эпоха = один шаг оптимизатора. Метрики считаются на эпохе 0
    и затем каждые cfg.eval_every эпох по всему train и test.
    """
    torch.set_num_threads(cfg.threads)
    if cfg.deterministic:
        enable_determinism()
    set_global_seed(0)  # глобальные генераторы не используются, но фиксируем на всякий случай
    device = resolve_device(cfg.device)
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

    data = make_modular_addition(cfg.data.q, cfg.data.train_frac, cfg.data.split_seed)
    X_tr, Y_tr, X_te, Y_te = (
        t.to(device=device, dtype=dtype) for t in (data.X_train, data.Y_train, data.X_test, data.Y_test)
    )
    model = BatchedMLP(cfg.seeds, 2 * cfg.data.q, cfg.model.d_hidden, cfg.data.q, cfg.model.n_hidden_layers)
    model = model.to(device=device, dtype=dtype)
    rho = torch.tensor(cfg.regularization.rho, dtype=dtype, device=device)
    optimizer = make_optimizer(model, cfg)
    kind = cfg.regularization.kind

    log: dict[str, list] = {k: [] for k in METRIC_KEYS}
    checkpoint_dir = Path(out_dir) / "checkpoints"

    def evaluate(step: int) -> None:
        with torch.no_grad():
            logits_tr, logits_te = model(X_tr), model(X_te)
            l1, l2sq = weight_norms(model)
            log["step"].append(step)
            log["train_loss"].append(cross_entropy_per_replica(logits_tr, Y_tr).cpu().numpy())
            log["test_loss"].append(cross_entropy_per_replica(logits_te, Y_te).cpu().numpy())
            log["train_acc"].append(accuracy_per_replica(logits_tr, Y_tr).cpu().numpy())
            log["test_acc"].append(accuracy_per_replica(logits_te, Y_te).cpu().numpy())
            log["penalty"].append(penalty(model, kind).cpu().numpy())
            log["l1_norm"].append(l1.cpu().numpy())
            log["l2_norm"].append(l2sq.sqrt().cpu().numpy())
            log["frac_small"].append(frac_small(model).cpu().numpy())
        if step in cfg.checkpoint_steps:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint_dir / f"step_{step}.pt")

    report_every = max(1, int(cfg.epochs * log_every_fraction))
    t0 = time.time()
    evaluate(0)
    for step in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        ce = cross_entropy_per_replica(model(X_tr), Y_tr)
        loss = (ce + rho * penalty(model, kind)).sum()
        loss.backward()
        optimizer.step()
        if step % cfg.eval_every == 0:
            evaluate(step)
        if step % report_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / step * (cfg.epochs - step)
            mean_test = float(np.mean(log["test_acc"][-1]))
            print(
                f"  эпоха {step}/{cfg.epochs}: {elapsed:.0f} c, осталось ~{eta:.0f} c, "
                f"средняя test_acc = {mean_test:.3f}",
                file=sys.stderr,
                flush=True,
            )
    elapsed = time.time() - t0

    metrics = {
        "rho": np.asarray(cfg.regularization.rho, dtype=np.float64),
        "seed": np.asarray(cfg.seeds, dtype=np.int64),
        **{k: np.asarray(v) for k, v in log.items()},
    }
    meta = {**environment_info(device), "elapsed_sec": round(elapsed, 1)}
    storage.save_job(Path(out_dir), cfg, metrics, meta)
    return metrics
