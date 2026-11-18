"""
Применить критерий момента перехода ко всем репликам и собрать detections.csv.

Пример:
    python scripts/detect.py --results results/lr_sweep results/grid_l2 results/grid_l1 \
        --out results/detections.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.detection import DetectionParams, detect
from grokking.storage import iter_jobs


def value_at(steps: np.ndarray, series: np.ndarray, t: int | None) -> float:
    """Значение метрики на записанной эпохе t; NaN, если t не определено."""
    if t is None:
        return float("nan")
    return float(series[int(np.searchsorted(steps, t))])


def detections_for_root(root: Path, params: DetectionParams) -> list[dict]:
    rows = []
    for job in iter_jobs(root):
        m = job.metrics
        steps = m["step"]
        for r in range(len(m["seed"])):
            d = detect(steps, m["train_acc"][:, r], m["test_acc"][:, r], params)
            rows.append(
                {
                    "experiment": job.config["experiment_name"],
                    "job_id": job.path.name,
                    "reg_kind": job.config["regularization"]["kind"],
                    "lr": job.config["optimizer"]["lr"],
                    "rho": float(m["rho"][r]),
                    "seed": int(m["seed"][r]),
                    "epochs": int(steps[-1]),
                    **d.as_row(),
                    # нормы весов в ключевые моменты — для гипотезы H5
                    "l2_norm_init": float(m["l2_norm"][0, r]),
                    "l2_norm_at_mem": value_at(steps, m["l2_norm"][:, r], d.t_mem),
                    "l2_norm_peak": float(m["l2_norm"][:, r].max()),
                    "l2_norm_at_grok": value_at(steps, m["l2_norm"][:, r], d.t_grok),
                    "l2_norm_end": float(m["l2_norm"][-1, r]),
                    "frac_small_at_mem": value_at(steps, m["frac_small"][:, r], d.t_mem),
                    "frac_small_end": float(m["frac_small"][-1, r]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Детекция момента перехода по всем репликам")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="каталоги экспериментов")
    parser.add_argument("--out", type=Path, required=True, help="куда сохранить detections.csv")
    args = parser.parse_args()

    params = DetectionParams()
    rows = [row for root in args.results for row in detections_for_root(root, params)]
    if not rows:
        raise SystemExit("не найдено ни одного завершённого задания")
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    counts = table.groupby(["experiment", "reg_kind", "outcome"]).size().unstack(fill_value=0)
    print(f"сохранено {len(table)} реплик: {args.out}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
