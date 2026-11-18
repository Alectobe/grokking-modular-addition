"""
Анализ результатов: таблицы и графики для п. 2 и п. 4 задания, проверка гипотез H1–H5
(docs/EXPERIMENTS.md §6) и ключевые числа для отчёта в results.json.

Пример:
    python scripts/analyze.py --detections results/detections.csv --results results \
        --figures report/figures --tables report/tables
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.plotting import (
    plot_end_accuracy,
    plot_lr_sweep,
    plot_norm_dynamics,
    plot_replica,
    plot_rho_sweep,
    plot_tgrok_vs,
    sci_label,
)
from grokking.report import save_json, save_table
from grokking.stats import (
    TRANSITION,
    compare_kinds_best,
    compare_kinds_completeness,
    compare_slopes,
    norm_tests,
    scaling_fit,
    spearman_trend,
    summarize,
)
from grokking.storage import iter_jobs

OUTCOME_NAMES = {
    "grokking": "grokking", "partial": "частичное", "no_generalization": "нет обобщения",
    "simultaneous": "без задержки", "no_memorization": "нет запоминания",
}


def find_replica(jobs, kind: str, lr: float, rho: float, seed: int):
    """Задание и номер реплики с заданными (тип, η, ρ, seed)."""
    for job in jobs:
        cfg = job.config
        if cfg["regularization"]["kind"] != kind or not np.isclose(cfg["optimizer"]["lr"], lr):
            continue
        hits = np.nonzero(np.isclose(job.metrics["rho"], rho) & (job.metrics["seed"] == seed))[0]
        if len(hits):
            return job, int(hits[0])
    raise LookupError(f"нет реплики {kind}, lr={lr}, rho={rho}, seed={seed}")


def det_row(det: pd.DataFrame, experiment: str, kind: str, lr: float, rho: float, seed: int):
    rows = det[(det.experiment == experiment) & (det.reg_kind == kind) & np.isclose(det.lr, lr)
               & np.isclose(det.rho, rho) & (det.seed == seed)]
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ результатов сетки")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="корень с каталогами экспериментов")
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    det = pd.read_csv(args.detections)
    for column in ("t_mem", "t_grok", "t_gen95"):
        det[column] = det[column].astype(float)
    figures, tables = args.figures, args.tables
    key_numbers: dict = {}

    # ---------- п. 2: подбор η ----------
    sweep_jobs = sorted(iter_jobs(args.results / "lr_sweep"), key=lambda j: j.config["optimizer"]["lr"])
    sweep = det[det.experiment == "lr_sweep"].sort_values(["lr", "seed"])
    q = sweep_jobs[0].config["data"]["q"]
    panels = []
    for job in sweep_jobs:
        lr = job.config["optimizer"]["lr"]
        r = int(np.nonzero(job.metrics["seed"] == 0)[0][0])
        panels.append((lr, job.metrics, r, det_row(det, "lr_sweep", "l2", lr, float(job.metrics["rho"][r]), 0)))
    plot_lr_sweep(panels, q, figures / "lr_sweep.png")
    sweep_table = sweep[["lr", "seed", "outcome", "t_mem", "t_grok", "end_test_acc"]].copy()
    sweep_table["outcome"] = sweep_table.outcome.map(OUTCOME_NAMES)
    sweep_table["t_grok / t_mem"] = sweep.t_grok / sweep.t_mem
    save_table(sweep_table.rename(columns={"lr": "η", "outcome": "исход", "end_test_acc": "итоговая test_acc"}),
               tables / "lr_sweep")
    key_numbers["lr_sweep"] = sweep[["lr", "seed", "outcome", "t_mem", "t_grok", "end_test_acc"]].to_dict("records")

    # ---------- кривые для иллюстраций ----------
    grid_jobs = list(iter_jobs(args.results / "grid_l2")) + list(iter_jobs(args.results / "grid_l1"))
    grid = det[det.experiment.isin(["grid_l2", "grid_l1"])].copy()
    job, r = find_replica(grid_jobs, "l2", 1e-3, 1e-4, 0)
    row = det_row(det, "grid_l2", "l2", 1e-3, 1e-4, 0)
    plot_replica(job.metrics, r, row.t_mem, row.t_grok, q,
                 f"Сложение по модулю {q}: L2, Adam, η = 1e-3, ρ = 1e-4, seed = 0", figures / "grokking_demo.png")
    for kind, name in (("l2", "L2"), ("l1", "L1")):
        job, _ = find_replica(grid_jobs, kind, 1e-3, float(grid[grid.reg_kind == kind].rho.max()), 0)
        plot_rho_sweep(job.metrics, 0, f"{name}, Adam, η = 1e-3, seed = 0: точность при разных ρ",
                       figures / f"rho_sweep_{kind}.png")

    # ---------- п. 4: сводка по ячейкам ----------
    summary = summarize(grid)
    summary_view = summary[["reg_kind", "lr", "rho", "n", "n_grokking", "n_partial", "n_no_generalization",
                            "n_simultaneous", "n_no_memorization", "t_mem_median", "t_grok_median", "t_grok_q25",
                            "t_grok_q75", "ratio_median", "share_gen95", "end_acc_median"]].copy()
    summary_view["reg_kind"] = summary_view.reg_kind.str.upper()
    save_table(summary_view.rename(columns={
        "reg_kind": "тип", "lr": "η", "rho": "ρ", "n": "реплик", "n_grokking": "grokking", "n_partial": "частичное",
        "n_no_generalization": "нет обобщ.", "n_simultaneous": "без задержки", "n_no_memorization": "нет запомин.",
        "t_mem_median": "t_mem (мед.)", "t_grok_median": "t_grok (мед.)", "t_grok_q25": "t_grok 25%",
        "t_grok_q75": "t_grok 75%", "ratio_median": "t_grok/t_mem", "share_gen95": "доля ≥95% test",
        "end_acc_median": "итог. test_acc"}), tables / "grid_summary")
    plot_tgrok_vs(summary, x="rho", by="lr", path=figures / "tgrok_vs_rho.png",
                  title="Момент перехода в зависимости от ρ (линии — η)")
    plot_tgrok_vs(summary, x="lr", by="rho", path=figures / "tgrok_vs_lr.png",
                  title="Момент перехода в зависимости от η (линии — ρ)")
    plot_end_accuracy(summary, figures / "end_accuracy.png")

    outcome_counts = grid.groupby(["reg_kind", "outcome"]).size().unstack(fill_value=0)
    key_numbers["outcomes"] = outcome_counts.to_dict("index")
    transitions = grid[grid.outcome.isin(TRANSITION) & (grid.rho > 0)]
    key_numbers["delay_ratio"] = {
        kind: {"median": float((part.t_grok / part.t_mem).median()), "min": float((part.t_grok / part.t_mem).min()),
               "max": float((part.t_grok / part.t_mem).max())}
        for kind, part in transitions.groupby("reg_kind")
    }
    seed_spread = transitions.groupby(["reg_kind", "lr", "rho"]).t_grok.agg(lambda s: s.max() / s.min())
    key_numbers["seed_spread_max_over_min"] = {k: float(v.median()) for k, v in seed_spread.groupby(level=0)}

    # ---------- гипотезы ----------
    h1 = spearman_trend(grid, "rho", ["reg_kind", "lr"])
    h2 = spearman_trend(grid, "lr", ["reg_kind", "rho"])
    for table, stem in ((h1, "h1_rho_trend"), (h2, "h2_lr_trend")):
        view = table.copy()
        view["reg_kind"] = view.reg_kind.str.upper()
        save_table(view.rename(columns={"reg_kind": "тип", "lr": "η", "rho": "ρ", "n": "реплик",
                                        "spearman_r": "r Спирмена", "p_holm": "p (Холм)"}), tables / stem)
    fits = {kind: scaling_fit(grid, kind) for kind in ("l2", "l1")}
    slopes = compare_slopes(fits["l2"], fits["l1"])
    fit_view = pd.DataFrame([{
        "тип": kind.upper(), "реплик": f["n"], "α (по η)": f["alpha"], "α 95% ДИ": f"[{f['alpha_low']:.2f}; {f['alpha_high']:.2f}]",
        "β (по ρ)": f["beta"], "β 95% ДИ": f"[{f['beta_low']:.2f}; {f['beta_high']:.2f}]", "R²": f["r2"],
        "p (α = β)": f["p_alpha_eq_beta"]} for kind, f in fits.items()])
    save_table(fit_view, tables / "h3_scaling")
    h4_best = compare_kinds_best(grid)
    h4_complete = compare_kinds_completeness(grid)
    h4_view = h4_best.copy()
    h4_view["rho_l2"] = h4_view.rho_l2.map(sci_label)
    h4_view["rho_l1"] = h4_view.rho_l1.map(sci_label)
    save_table(h4_view.rename(columns={
        "lr": "η", "rho_l2": "лучшее ρ (L2)", "rho_l1": "лучшее ρ (L1)", "t_grok_l2": "t_grok L2",
        "t_grok_l1": "t_grok L1", "ratio_l1_to_l2": "L1/L2", "ratio_low": "L1/L2 ДИ низ", "ratio_high": "L1/L2 ДИ верх",
        "cliffs_delta_l1_vs_l2": "δ Клиффа", "p_holm": "p (Холм)"}), tables / "h4_best_cells")
    h5 = norm_tests(grid)
    save_table(h5.rename(columns={"test": "сравнение", "n": "реплик", "median_before": "медиана до",
                                  "median_after": "медиана после", "median_change": "после/до",
                                  "p_holm": "p (Холм)"}), tables / "h5_norms")

    # Устойчивость выводов к выбору критерия: те же H1 и H3 с альтернативными оценками t_grok
    robust_rows = []
    for column, name in (("t_grok", "ступенька (основной)"), ("t_grok_thr50", "порог середины"),
                         ("t_grok_logderiv", "максимум производной")):
        alt = grid.copy()
        alt["t_grok"] = alt[column].astype(float)
        h1_alt = spearman_trend(alt, "rho", ["reg_kind", "lr"])
        for kind in ("l2", "l1"):
            fit = scaling_fit(alt, kind)
            part = h1_alt[h1_alt.reg_kind == kind]
            robust_rows.append({
                "критерий": name, "тип": kind.upper(), "α (по η)": fit["alpha"], "β (по ρ)": fit["beta"],
                "H1: значимых групп (p Холм < 0.05)": f"{int((part.p_holm < 0.05).sum())} из {len(part)}",
                "H1: знак r": "все < 0" if (part.spearman_r < 0).all() else "разный",
            })
    robust = pd.DataFrame(robust_rows)
    save_table(robust, tables / "robustness_criteria")

    rows = []
    for label, kind, lr, rho in (("L2, ρ = 1e-4: grokking", "l2", 1e-3, 1e-4), ("Контроль ρ = 0: нет обобщения", "l2", 1e-3, 0.0),
                                 ("L1, ρ = 3e-6: частичное обобщение", "l1", 1e-3, 3e-6)):
        job, r = find_replica(grid_jobs, kind, lr, rho, 0)
        rows.append((label + " (η = 1e-3, seed 0)", job.metrics, r,
                     det_row(det, f"grid_{kind}", kind, lr, rho, 0)))
    plot_norm_dynamics(rows, figures / "norm_dynamics.png")

    key_numbers.update({
        "h1": h1.to_dict("records"), "h2": h2.to_dict("records"), "h3": fits, "h3_slope_difference": slopes,
        "h4_best": h4_best.to_dict("records"), "h4_completeness": h4_complete, "h5": h5.to_dict("records"),
        "n_replicas_grid": len(grid), "n_replicas_sweep": len(sweep), "robustness": robust.to_dict("records"),
    })
    save_json(key_numbers, tables / "results.json")
    print(summary_view.to_string(index=False))
    for name, table in (("H1", h1), ("H2", h2), ("H3", fit_view), ("H4", h4_best), ("H5", h5),
                        ("Устойчивость к критерию", robust)):
        print(f"\n{name}\n{table.to_string(index=False)}")
    print("\nH4 полнота:", h4_complete, "\nH4 наклоны:", slopes)


if __name__ == "__main__":
    main()
