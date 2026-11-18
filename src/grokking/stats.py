"""
Статистический анализ таблицы детекций (docs/EXPERIMENTS.md §5–6).

Единица наблюдения — реплика. Ячейка сетки — (тип регуляризации, η, ρ).
Отклик t_grok наблюдается у исходов grokking и partial; у no_generalization он
цензурирован справа: известно только, что t_grok > T (T — число эпох).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRANSITION = ("grokking", "partial")  # исходы, у которых t_grok наблюдается
AT_RISK = ("grokking", "partial", "no_generalization")  # запомнили train и могли перейти


# ---------- описательные величины с учётом цензурирования ----------


def km_quantile(times: np.ndarray, observed: np.ndarray, q: float) -> float:
    """Квантиль q времени до события по Каплану–Мейеру; inf, если доля событий меньше q.

    Без цензурирования возвращает обычный выборочный квантиль (np.quantile).
    """
    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if len(times) == 0:
        return math.nan
    if observed.all():
        return float(np.quantile(times, q))
    if not observed.any():
        return math.inf
    data = stats.CensoredData(uncensored=times[observed], right=times[~observed])
    sf = stats.ecdf(data).sf
    reached = np.nonzero(sf.probabilities <= 1.0 - q)[0]
    return float(sf.quantiles[reached[0]]) if len(reached) else math.inf


def transition_times(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """t_grok реплик, которые могли перейти: наблюдённые и цензурированные на T."""
    at_risk = group[group.outcome.isin(AT_RISK)]
    observed = at_risk.outcome.isin(TRANSITION).to_numpy()
    times = np.where(observed, at_risk.t_grok.to_numpy(dtype=float), at_risk.epochs.to_numpy(dtype=float))
    return times, observed


def gen95_times(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """t_gen95 реплик, запомнивших train: наблюдённые и цензурированные на T."""
    memorized = group[group.t_mem.notna()]
    observed = ~memorized.censored_gen95.to_numpy(dtype=bool)
    times = np.where(observed, memorized.t_gen95.to_numpy(dtype=float), memorized.epochs.to_numpy(dtype=float))
    return times, observed


def summarize(det: pd.DataFrame, by: tuple[str, ...] = ("reg_kind", "lr", "rho")) -> pd.DataFrame:
    """Сводка по ячейкам: доли исходов, медианы и квартили t_grok (Каплан–Мейер), полнота обобщения."""
    rows = []
    for key, group in det.groupby(list(by), sort=True):
        times, observed = transition_times(group)
        g_times, g_observed = gen95_times(group)
        ratio = (group.t_grok / group.t_mem)[group.outcome.isin(TRANSITION)]
        counts = group.outcome.value_counts()
        rows.append(
            {
                **dict(zip(by, key)),
                "n": len(group),
                **{f"n_{o}": int(counts.get(o, 0)) for o in
                   ("grokking", "partial", "no_generalization", "simultaneous", "no_memorization")},
                "t_mem_median": float(group.t_mem.median()),
                "t_grok_median": km_quantile(times, observed, 0.5) if observed.any() else math.nan,
                "t_grok_q25": km_quantile(times, observed, 0.25) if observed.any() else math.nan,
                "t_grok_q75": km_quantile(times, observed, 0.75) if observed.any() else math.nan,
                "ratio_median": float(ratio.median()) if len(ratio) else math.nan,
                "t_gen95_median": km_quantile(g_times, g_observed, 0.5) if len(g_times) else math.nan,
                "share_gen95": float(g_observed.mean()) if len(g_observed) else math.nan,
                "end_acc_median": float(group.end_test_acc.median()),
                "end_acc_q25": float(group.end_test_acc.quantile(0.25)),
                "end_acc_q75": float(group.end_test_acc.quantile(0.75)),
                "l2_norm_end_median": float(group.l2_norm_end.median()),
                "frac_small_end_median": float(group.frac_small_end.median()),
            }
        )
    return pd.DataFrame(rows)


# ---------- проверка гипотез ----------


def holm(pvalues) -> np.ndarray:
    """Поправка Холма–Бонферрони; возвращает скорректированные p-значения в исходном порядке."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def cliffs_delta(a, b) -> float:
    """δ Клиффа: P(a > b) − P(a < b), от −1 до 1; inf сравниваются как обычные числа."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (len(a) * len(b)))


def bootstrap_median_ratio(a, b, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Отношение медиан median(a) / median(b) и его 95% перцентильный бутстреп-интервал."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    boot_a = np.median(rng.choice(a, size=(n_boot, len(a))), axis=1)
    boot_b = np.median(rng.choice(b, size=(n_boot, len(b))), axis=1)
    ratios = boot_a / boot_b
    return float(np.median(a) / np.median(b)), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def spearman_trend(det: pd.DataFrame, x: str, within: list[str]) -> pd.DataFrame:
    """H1/H2: ранговая корреляция log x и t_grok внутри групп (цензурированные — как +inf).

    Отрицательный коэффициент — t_grok убывает с ростом x. p-значения поправлены по Холму
    внутри таблицы. Контроль ρ = 0 не участвует: на логарифмической шкале его нет.
    """
    rows = []
    sub = det[(det.rho > 0) & det.outcome.isin(AT_RISK)]
    for key, group in sub.groupby(within, sort=True):
        if group[x].nunique() < 2:
            continue
        t = np.where(group.outcome.isin(TRANSITION), group.t_grok.astype(float), math.inf)
        result = stats.spearmanr(np.log10(group[x].astype(float)), t)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(within, key)), "n": len(group), "spearman_r": float(result.statistic),
                     "p": float(result.pvalue)})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table.p)
    return table


def scaling_fit(det: pd.DataFrame, kind: str) -> dict:
    """H3: МНК log10 t_grok = c − α·log10 η − β·log10 ρ по наблюдённым переходам (ρ > 0).

    Возвращает α и β с 95% ДИ, R² и p-значение проверки α = β (t-тест разности коэффициентов).
    """
    sub = det[(det.reg_kind == kind) & (det.rho > 0) & det.outcome.isin(TRANSITION)]
    y = np.log10(sub.t_grok.to_numpy(dtype=float))
    X = np.column_stack([np.ones(len(sub)), np.log10(sub.lr.to_numpy(dtype=float)),
                         np.log10(sub.rho.to_numpy(dtype=float))])
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_crit = stats.t.ppf(0.975, dof)
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    diff = beta[1] - beta[2]
    se_diff = math.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
    p_equal = 2 * stats.t.sf(abs(diff / se_diff), dof)
    return {
        "reg_kind": kind,
        "n": n,
        "alpha": -beta[1], "alpha_low": -beta[1] - t_crit * se[1], "alpha_high": -beta[1] + t_crit * se[1],
        "beta": -beta[2], "beta_low": -beta[2] - t_crit * se[2], "beta_high": -beta[2] + t_crit * se[2],
        "beta_se": se[2],
        "r2": r2,
        "p_alpha_eq_beta": p_equal,
    }


def best_cells(det: pd.DataFrame, min_share: float = 0.8) -> pd.DataFrame:
    """Для каждой пары (тип, η) — ячейка ρ с наименьшей медианой t_grok среди ячеек,
    где переход (grokking или partial) произошёл хотя бы у min_share реплик."""
    summary = summarize(det[det.rho > 0])
    summary["share_transition"] = (summary.n_grokking + summary.n_partial) / summary.n
    eligible = summary[summary.share_transition >= min_share]
    idx = eligible.groupby(["reg_kind", "lr"]).t_grok_median.idxmin()
    return eligible.loc[idx, ["reg_kind", "lr", "rho", "t_grok_median"]].reset_index(drop=True)


def compare_kinds_best(det: pd.DataFrame) -> pd.DataFrame:
    """H4(1): лучшее достижимое t_grok при каждом η — L2 против L1 (Манн–Уитни, Холм, δ Клиффа)."""
    best = best_cells(det)
    rows = []
    for lr in sorted(best.lr.unique()):
        cells = best[best.lr == lr].set_index("reg_kind")
        if not {"l1", "l2"} <= set(cells.index):
            continue
        samples = {}
        for kind in ("l2", "l1"):
            part = det[(det.reg_kind == kind) & (det.lr == lr) & (det.rho == cells.loc[kind, "rho"])]
            samples[kind] = np.where(part.outcome.isin(TRANSITION), part.t_grok.astype(float), math.inf)
        test = stats.mannwhitneyu(samples["l2"], samples["l1"], alternative="two-sided")
        ratio, low, high = bootstrap_median_ratio(samples["l1"], samples["l2"])
        rows.append({
            "lr": lr, "rho_l2": cells.loc["l2", "rho"], "rho_l1": cells.loc["l1", "rho"],
            "t_grok_l2": float(np.median(samples["l2"])), "t_grok_l1": float(np.median(samples["l1"])),
            "ratio_l1_to_l2": ratio, "ratio_low": low, "ratio_high": high,
            "cliffs_delta_l1_vs_l2": cliffs_delta(samples["l1"], samples["l2"]), "p": float(test.pvalue),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table.p)
    return table


def compare_kinds_completeness(det: pd.DataFrame) -> dict:
    """H4(2): доля реплик с полным обобщением (t_gen95 достигнут) у L2 и L1, точный тест Фишера."""
    sub = det[det.rho > 0]
    table = []
    for kind in ("l2", "l1"):
        part = sub[sub.reg_kind == kind]
        full = int((~part.censored_gen95.astype(bool)).sum())
        table.append([full, len(part) - full])
    result = stats.fisher_exact(table)
    return {
        "l2_full": table[0][0], "l2_total": sum(table[0]), "l1_full": table[1][0], "l1_total": sum(table[1]),
        "odds_ratio": float(result.statistic), "p": float(result.pvalue),
    }


def compare_slopes(fit_a: dict, fit_b: dict) -> dict:
    """H4(3): разность наклонов β двух независимых регрессий (z-тест)."""
    diff = fit_a["beta"] - fit_b["beta"]
    se = math.sqrt(fit_a["beta_se"] ** 2 + fit_b["beta_se"] ** 2)
    return {"beta_diff": diff, "z": diff / se, "p": float(2 * stats.norm.sf(abs(diff / se)))}


def norm_tests(det: pd.DataFrame) -> pd.DataFrame:
    """H5: что происходит с весами. Парные тесты Уилкоксона и Манн–Уитни, поправка Холма по семейству."""
    rows = []
    l2_grok = det[(det.reg_kind == "l2") & (det.rho > 0) & (det.outcome == "grokking")]
    control = det[(det.reg_kind == "l2") & (det.rho == 0) & det.t_mem.notna()]
    l1_trans = det[(det.reg_kind == "l1") & det.outcome.isin(TRANSITION)]
    l1_all, l2_all = det[(det.reg_kind == "l1") & (det.rho > 0)], det[(det.reg_kind == "l2") & (det.rho > 0)]

    def paired(name, before, after):
        result = stats.wilcoxon(after, before)
        rows.append({"test": name, "n": len(before), "median_before": float(np.median(before)),
                     "median_after": float(np.median(after)), "median_change": float(np.median(after / before)),
                     "p": float(result.pvalue)})

    paired("L2 с grokking: L2-норма, пик → конец", l2_grok.l2_norm_peak.to_numpy(), l2_grok.l2_norm_end.to_numpy())
    paired("L2 с grokking: L2-норма, t_mem → конец", l2_grok.l2_norm_at_mem.to_numpy(), l2_grok.l2_norm_end.to_numpy())
    paired("Контроль ρ = 0: L2-норма, t_mem → конец", control.l2_norm_at_mem.to_numpy(), control.l2_norm_end.to_numpy())
    paired("L1 с переходом: доля |w| < 0.001, t_mem → конец", l1_trans.frac_small_at_mem.to_numpy(),
           l1_trans.frac_small_end.to_numpy())
    result = stats.mannwhitneyu(l1_all.frac_small_end, l2_all.frac_small_end, alternative="two-sided")
    rows.append({"test": "Доля |w| < 0.001 в конце: L1 против L2", "n": len(l1_all) + len(l2_all),
                 "median_before": float(l2_all.frac_small_end.median()),
                 "median_after": float(l1_all.frac_small_end.median()),
                 "median_change": float(l1_all.frac_small_end.median() / l2_all.frac_small_end.median()),
                 "p": float(result.pvalue)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p)
    return table
