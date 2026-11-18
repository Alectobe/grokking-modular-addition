"""Служебные функции: seed, детерминизм, устройство, сведения об окружении."""
from __future__ import annotations

import os
import platform
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_global_seed(seed: int) -> None:
    """Фиксирует глобальные генераторы random, numpy и torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def enable_determinism() -> None:
    """Детерминированные алгоритмы PyTorch; для cuBLAS нужна переменная окружения."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def resolve_device(name: str) -> torch.device:
    """'auto' — CUDA, если она есть, иначе CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("в конфиге device: cuda, но CUDA недоступна")
    return torch.device(name)


def git_commit() -> str | None:
    """Хеш текущего коммита и отметка о незакоммиченных изменениях; None вне git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


def environment_info(device: torch.device) -> dict:
    """Версии библиотек, git-коммит и устройство — пишутся в meta.json задания."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "threads": torch.get_num_threads(),
        "git_commit": git_commit(),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
