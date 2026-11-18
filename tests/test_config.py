from pathlib import Path

import pytest
import yaml

from grokking.config import job_id, load_config, load_experiment
from grokking.grid import shard

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_base_config_is_one_job():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.n_replicas == len(cfg.seeds)
    assert len(set(cfg.regularization.rho)) == 1


def test_grid_expansion_and_replica_order():
    jobs = load_experiment(CONFIGS / "grid_l2.yaml")
    assert len(jobs) == 3
    assert {cfg.optimizer.lr for cfg in jobs} == {1e-3, 2e-3, 4e-3}
    cfg = jobs[0]
    n_seeds = len(set(cfg.seeds))
    assert cfg.n_replicas == 6 * n_seeds
    # реплики идут блоками по rho, внутри блока — все seed
    assert cfg.regularization.rho[:n_seeds] == (0.0,) * n_seeds
    assert cfg.seeds[:n_seeds] == tuple(sorted(set(cfg.seeds)))
    assert len({job_id(c) for c in jobs}) == 3


def test_shards_partition_jobs():
    jobs = [cfg for name in ["grid_l1.yaml", "grid_l2.yaml", "lr_sweep.yaml"] for cfg in load_experiment(CONFIGS / name)]
    parts = [shard(jobs, i, 3) for i in (1, 2, 3)]
    ids = [job_id(cfg) for part in parts for cfg in part]
    assert sorted(ids) == sorted(job_id(cfg) for cfg in jobs)
    assert len(ids) == len(set(ids))


def write_experiment(tmp_path, **sections) -> Path:
    base = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({"base": "base.yaml", **sections}), encoding="utf-8")
    return path


def test_unknown_override_rejected(tmp_path):
    path = write_experiment(tmp_path, overrides={"optimizer.learning_rate": 0.1})
    with pytest.raises(ValueError):
        load_experiment(path)


def test_lr_cannot_vary_between_replicas(tmp_path):
    path = write_experiment(tmp_path, grid={"replicas_over": {"optimizer.lr": [1e-3, 2e-3]}})
    with pytest.raises(ValueError):
        load_experiment(path)


def test_non_prime_q_rejected(tmp_path):
    path = write_experiment(tmp_path, overrides={"data.q": 100})
    with pytest.raises(ValueError):
        load_experiment(path)
