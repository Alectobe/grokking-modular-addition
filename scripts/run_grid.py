"""
Обучить все задания одного или нескольких экспериментов. Готовые задания пропускаются,
поэтому после падения достаточно запустить команду ещё раз.

Примеры:
    python scripts/run_grid.py --grid configs/lr_sweep.yaml configs/grid_l2.yaml configs/grid_l1.yaml
    python scripts/run_grid.py --grid configs/grid_l2.yaml --shard 1/3   # треть заданий
    python scripts/run_grid.py --grid configs/grid_l1.yaml --dry-run     # только список заданий
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.config import job_id  # noqa: E402
from grokking.grid import expand_grid, job_dir, pending_jobs, shard  # noqa: E402
from grokking.train import train_job  # noqa: E402


def parse_shard(text: str) -> tuple[int, int]:
    index, count = (int(part) for part in text.split("/"))
    return index, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск сетки экспериментов")
    parser.add_argument("--grid", type=Path, nargs="+", required=True, help="YAML экспериментов")
    parser.add_argument("--shard", type=parse_shard, default=(1, 1), help="часть i/n, например 1/3")
    parser.add_argument("--dry-run", action="store_true", help="только показать задания")
    args = parser.parse_args()

    jobs = [cfg for path in args.grid for cfg in expand_grid(path)]
    # Состав части определяется по job_id, а порядок выполнения — порядком конфигов в --grid
    selected_ids = {job_id(cfg) for cfg in shard(jobs, *args.shard)}
    selected = [cfg for cfg in jobs if job_id(cfg) in selected_ids]
    todo = pending_jobs(selected)
    replicas = sum(cfg.n_replicas for cfg in todo)
    print(f"заданий всего {len(jobs)}, в части {len(selected)}, осталось {len(todo)} ({replicas} реплик)", flush=True)

    for i, cfg in enumerate(todo, start=1):
        line = (f"[{i}/{len(todo)}] {cfg.experiment_name}/{job_id(cfg)}: {cfg.regularization.kind}, "
                f"lr={cfg.optimizer.lr:g}, rho={sorted(set(cfg.regularization.rho))}, {cfg.n_replicas} реплик")
        print(line, flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        train_job(cfg, job_dir(cfg))
        print(f"    готово за {time.time() - t0:.0f} c", flush=True)


if __name__ == "__main__":
    main()
