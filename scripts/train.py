"""
Обучить одно задание по конфигу.

Пример:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --out results/demo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.config import job_id, load_config  # noqa: E402
from grokking.grid import job_dir  # noqa: E402
from grokking.train import train_job  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение одного задания")
    parser.add_argument("--config", type=Path, required=True, help="YAML с одним заданием")
    parser.add_argument("--out", type=Path, default=None, help="каталог результатов (по умолчанию из конфига)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = args.out if args.out is not None else job_dir(cfg)
    print(f"{job_id(cfg)}: {cfg.n_replicas} реплик, {cfg.epochs} эпох -> {out}", flush=True)
    train_job(cfg, out)
    print(f"готово: {out}")


if __name__ == "__main__":
    main()
