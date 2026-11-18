"""Разворачивание сетки в задания, пропуск готовых, разбиение между машинами."""
from __future__ import annotations

from pathlib import Path

from . import storage
from .config import Config, job_id, load_experiment


def expand_grid(path: str | Path) -> list[Config]:
    """Все задания эксперимента: одно задание = одна комбинация grid.jobs_over."""
    return load_experiment(path)


def job_dir(cfg: Config) -> Path:
    """Каталог результатов задания: <output_root>/<experiment>/<job_id>."""
    return Path(cfg.output_root) / cfg.experiment_name / job_id(cfg)


def pending_jobs(jobs: list[Config]) -> list[Config]:
    """Задания, у которых ещё нет meta.json со статусом done."""
    return [cfg for cfg in jobs if not storage.is_done(job_dir(cfg))]


def shard(jobs: list[Config], index: int, count: int) -> list[Config]:
    """Часть index из count (index от 1 до count), детерминированно по job_id.

    Части не пересекаются и вместе дают все задания, поэтому их можно считать
    на разных машинах и потом просто объединить каталоги results/.
    """
    if not 1 <= index <= count:
        raise ValueError(f"номер части {index} вне диапазона 1..{count}")
    ordered = sorted(jobs, key=job_id)
    return ordered[index - 1 :: count]
