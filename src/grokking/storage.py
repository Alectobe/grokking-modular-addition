"""Запись и чтение результатов одного задания: config.yaml, metrics.npz, meta.json.

Запись атомарная: каждый файл сначала пишется во временный и переименовывается.
meta.json пишется последним, поэтому задание без него считается незавершённым.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import Config


@dataclass
class JobRecord:
    path: Path
    config: dict
    metrics: dict[str, np.ndarray]
    meta: dict


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_job(out_dir: Path, cfg: Config, metrics: dict[str, np.ndarray], meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_dir / "config.yaml", yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False))
    tmp = out_dir / "metrics.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **metrics)
    os.replace(tmp, out_dir / "metrics.npz")
    _atomic_write_text(out_dir / "meta.json", json.dumps({**meta, "status": "done"}, indent=2, ensure_ascii=False))


def is_done(job_dir: Path) -> bool:
    meta_path = Path(job_dir) / "meta.json"
    if not meta_path.exists():
        return False
    return json.loads(meta_path.read_text(encoding="utf-8")).get("status") == "done"


def load_job(job_dir: Path) -> JobRecord:
    job_dir = Path(job_dir)
    with np.load(job_dir / "metrics.npz") as npz:
        metrics = {k: npz[k] for k in npz.files}
    return JobRecord(
        path=job_dir,
        config=yaml.safe_load((job_dir / "config.yaml").read_text(encoding="utf-8")),
        metrics=metrics,
        meta=json.loads((job_dir / "meta.json").read_text(encoding="utf-8")),
    )


def iter_jobs(root: Path) -> Iterator[JobRecord]:
    """Все завершённые задания в каталоге root (на любой глубине), в порядке путей."""
    for meta_path in sorted(Path(root).rglob("meta.json")):
        if is_done(meta_path.parent):
            yield load_job(meta_path.parent)
