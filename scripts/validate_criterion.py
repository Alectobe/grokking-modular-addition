"""
Проверка критерия момента перехода (п. 3 задания, docs/EXPERIMENTS.md §3.3).

1. Синтетические кривые с известным t*: ошибка |log10(t̂ / t*)| основного критерия
   и альтернатив (порог середины, максимум производной по log t, середина сигмоиды).
2. Ложные срабатывания: синтетические кривые без перехода.
3. Реальные запуски сетки: согласие критериев и устойчивость к прореживанию записи метрик.

Пример:
    python scripts/validate_criterion.py --detections results/detections.csv \
        --results results/grid_l2 results/grid_l1 --figures report/figures --tables report/tables
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.detection import (
    DetectionParams,
    detect,
    max_log_derivative,
    sigmoid_midpoint,
    synthetic_curves,
    threshold_crossing,
)
from grokking.plotting import plot_criterion_agreement, plot_criterion_errors, plot_criterion_example
from grokking.report import save_json, save_table
from grokking.stats import TRANSITION
from grokking.storage import iter_jobs

STEPS = np.arange(0, 6001, 10)
METHODS = ("ступенька (основной)", "порог середины", "максимум производной", "середина сигмоиды")


def estimate_all(steps, train, test, params) -> tuple[object, dict[str, float | None]]:
    """Детекция и оценки момента перехода всеми критериями (None — критерий не дал ответа)."""
    d = detect(steps, train, test, params)
    if d.outcome not in TRANSITION:
        return d, dict.fromkeys(METHODS)
    i_mem = int(np.searchsorted(steps, d.t_mem))
    tail = test[-max(1, round(params.tail_fraction * len(test))):].mean()
    midpoint = 0.5 * (test[i_mem] + tail)
    return d, {
        METHODS[0]: d.t_grok,
        METHODS[1]: threshold_crossing(steps, test, i_mem, midpoint),
        METHODS[2]: max_log_derivative(steps, test, i_mem, params.smooth_window, params.log_grid_points),
        METHODS[3]: sigmoid_midpoint(steps, test, i_mem, d.t_grok),
    }


def synthetic_study(n: int, seed: int, params: DetectionParams):
    """Случайные кривые с переходом: ошибки критериев по условиям."""
    rng = np.random.default_rng(seed)
    rows, example = [], None
    for i in range(n):
        t_mem = rng.uniform(100, 500)
        t_star = min(t_mem * math.exp(rng.uniform(math.log(1.5), math.log(10))), 5000.0)
        width = rng.uniform(0.05, 0.35)
        high = rng.uniform(0.35, 1.0)
        noise = rng.uniform(0.0, 0.03)
        dip = None
        if rng.random() < 0.3:
            dip = (t_star * rng.uniform(1.2, 2.5), rng.uniform(50, 400), rng.uniform(0.1, 0.4))
        train, test = synthetic_curves(STEPS, t_mem, t_star, width, rng.uniform(0, 0.02), high, noise, rng, dip)
        d, est = estimate_all(STEPS, train, test, params)
        if i == 0:
            example = (train, test, t_star, est)
        for method, value in est.items():
            rows.append({
                "curve": i, "method": method, "t_star": t_star, "estimate": value,
                "error": abs(math.log10(value / t_star)) if value else math.nan,
                "late": t_star > 1500, "dip": dip is not None, "partial": high < 0.95, "high": high,
                "detected": d.outcome in TRANSITION,
            })
    return pd.DataFrame(rows), example


def false_positive_study(n: int, seed: int, params: DetectionParams) -> float:
    """Кривые без перехода: шум около нуля и медленный дрейф до 10%. Доля ложных переходов."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        train, _ = synthetic_curves(STEPS, rng.uniform(100, 500), 1000.0, 0.1, 0.0, 1.0, 0.0, rng)
        drift = rng.uniform(0, 0.1) * STEPS / STEPS[-1]
        test = np.clip(rng.uniform(0, 0.02) + drift + rng.normal(0, rng.uniform(0, 0.03), len(STEPS)), 0, 1)
        hits += detect(STEPS, train, test, params).outcome in TRANSITION
    return hits / n


def real_study(roots: list[Path], params: DetectionParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оценки всех критериев и t_grok при прореживании записи для реальных реплик."""
    rows, thinning = [], []
    for root in roots:
        for job in iter_jobs(root):
            m = job.metrics
            kind = job.config["regularization"]["kind"]
            for r in range(len(m["seed"])):
                d, est = estimate_all(m["step"], m["train_acc"][:, r], m["test_acc"][:, r], params)
                if d.outcome not in TRANSITION:
                    continue
                rows.append({"reg_kind": kind, **est})
                for k in (2, 5):
                    d_k = detect(m["step"][::k], m["train_acc"][::k, r], m["test_acc"][::k, r], params)
                    thinning.append({
                        "reg_kind": kind, "thin": k, "same_outcome": d_k.outcome == d.outcome,
                        "error": abs(math.log10(d_k.t_grok / d.t_grok)) if d_k.t_grok else math.nan,
                    })
    return pd.DataFrame(rows), pd.DataFrame(thinning)


def threshold_sensitivity(roots: list[Path], jumps=(0.2, 0.3, 0.4)) -> pd.DataFrame:
    """Исходы реплик сетки при разных порогах скачка δ_min (остальные параметры по умолчанию)."""
    rows = []
    for root in roots:
        for job in iter_jobs(root):
            m = job.metrics
            kind = job.config["regularization"]["kind"]
            for r in range(len(m["seed"])):
                outcomes = {j: detect(m["step"], m["train_acc"][:, r], m["test_acc"][:, r],
                                      DetectionParams(min_jump=j)).outcome for j in jumps}
                rows.append({"reg_kind": kind, "rho": float(m["rho"][r]), **{f"d{j}": o for j, o in outcomes.items()}})
    table = pd.DataFrame(rows)
    table = table[table.rho > 0]
    result = []
    for kind, part in table.groupby("reg_kind"):
        for j in jumps:
            column = part[f"d{j}"]
            result.append({
                "тип": kind.upper(), "порог δ_min": j, "реплик": len(part),
                "grokking": int((column == "grokking").sum()), "частичное": int((column == "partial").sum()),
                "нет обобщения": int((column == "no_generalization").sum()),
                "исход отличается от δ_min = 0.3, доля": float((column != part["d0.3"]).mean()),
            })
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка критерия момента перехода")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--n-synthetic", type=int, default=600)
    args = parser.parse_args()
    params = DetectionParams()

    synth_all, _ = synthetic_study(args.n_synthetic, seed=0, params=params)
    # Наглядный пример: полный переход, шум и провал после перехода (на провалах ошибается производная)
    rng = np.random.default_rng(7)
    train, test = synthetic_curves(STEPS, 300.0, 1500.0, 0.2, 0.0, 1.0, 0.02, rng, dip=(2600.0, 250.0, 0.35))
    _, est = estimate_all(STEPS, train, test, params)
    plot_criterion_example(STEPS, train, test, 1500.0, est, args.figures / "criterion_example.png")

    # Доля распознанных переходов зависит от величины скачка: скачок < 0.3 по определению не переход
    curves = synth_all[synth_all.method == METHODS[0]]
    bins = pd.cut(curves.high, [0.35, 0.5, 0.7, 0.95, 1.0], include_lowest=True)
    recognition = curves.groupby(bins, observed=True).detected.agg(["size", "mean"]).reset_index()
    recognition.columns = ["итоговый уровень test_acc (high)", "кривых", "распознан переход, доля"]
    recognition["итоговый уровень test_acc (high)"] = recognition["итоговый уровень test_acc (high)"].astype(str)
    save_table(recognition, args.tables / "criterion_recognition")

    # Точность локализации — только на распознанных кривых
    synth = synth_all[synth_all.detected]
    errors = {m: synth[(synth.method == m) & synth.error.notna()].error.to_numpy() for m in METHODS}
    plot_criterion_errors(errors, args.figures / "criterion_errors.png")

    def describe(part: pd.DataFrame) -> dict:
        ok = part.error.dropna()
        return {"кривых": len(part), "медиана ошибки": ok.median(), "95-й перцентиль": ok.quantile(0.95),
                "в пределах 10%, доля": (ok < 0.04).sum() / len(part), "нет ответа, доля": part.error.isna().mean()}

    conditions = [("все", slice(None)), ("переход после 1500-й эпохи", "late"), ("с провалом после перехода", "dip"),
                  ("неполное обобщение (high < 0.95)", "partial")]
    synth_rows = []
    for method in METHODS:
        part = synth[synth.method == method]
        for name, column in conditions:
            sub = part if name == "все" else part[part[column]]
            synth_rows.append({"критерий": method, "условие": name, **describe(sub)})
    save_table(pd.DataFrame(synth_rows), args.tables / "criterion_synthetic")
    fp_rate = false_positive_study(300, seed=1, params=params)

    det = pd.read_csv(args.detections)
    grid = det[det.experiment.isin(["grid_l1", "grid_l2"])]
    plot_criterion_agreement(grid[grid.outcome.isin(TRANSITION)], args.figures / "criterion_agreement.png")
    real, thinning = real_study(args.results, params)
    agree_rows = []
    for method in METHODS[1:]:
        for kind in ("l2", "l1"):
            part = real[(real.reg_kind == kind)].dropna(subset=[METHODS[0], method])
            ratio = np.abs(np.log10(part[method].astype(float) / part[METHODS[0]].astype(float)))
            agree_rows.append({
                "альтернатива": method, "тип": kind.upper(), "реплик": len(part),
                "Спирмен с основным": stats.spearmanr(part[METHODS[0]], part[method]).statistic,
                "медиана |log10 отношения|": ratio.median(), "95-й перцентиль": ratio.quantile(0.95),
                "нет ответа, доля": 1 - len(part) / max(1, (real.reg_kind == kind).sum()),
            })
    save_table(pd.DataFrame(agree_rows), args.tables / "criterion_agreement")
    thin_rows = []
    for (kind, k), part in thinning.groupby(["reg_kind", "thin"]):
        thin_rows.append({"тип": kind.upper(), "прореживание": f"каждая {k}-я точка", "реплик": len(part),
                          "исход не изменился, доля": part.same_outcome.mean(),
                          "медиана |log10(t_grok / t_grok полн.)|": part.error.median(),
                          "максимум": part.error.max()})
    save_table(pd.DataFrame(thin_rows), args.tables / "criterion_thinning")
    sensitivity = threshold_sensitivity(args.results)
    save_table(sensitivity, args.tables / "criterion_threshold_sensitivity")

    main_errors = synth[synth.method == METHODS[0]].error
    summary = {
        "synthetic_curves": args.n_synthetic,
        "synthetic_recognized_share": float(curves.detected.mean()),
        "main_median_error": float(main_errors.median()),
        "main_p95_error": float(main_errors.quantile(0.95)),
        "main_share_within_10pct": float((main_errors < 0.04).mean()),
        "false_positive_rate": fp_rate,
        "real_replicas_with_transition": len(real),
    }
    save_json(summary, args.tables / "criterion_summary.json")
    print(recognition.to_string(index=False))
    print(pd.DataFrame(synth_rows).to_string(index=False))
    print(pd.DataFrame(agree_rows).to_string(index=False))
    print(pd.DataFrame(thin_rows).to_string(index=False))
    print(sensitivity.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
