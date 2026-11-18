"""
Графики проекта: единый стиль и читаемые оси.

Соглашения:
  * эпохи — логарифмическая шкала с подписями обычными числами (50, 1 000, 5 000),
    эпоха 0 не рисуется: на логарифмической оси её нельзя показать;
  * точность — в процентах; перекрёстная энтропия — логарифмическая шкала с уровнем
    случайного угадывания ln q;
  * упорядоченные величины (η, ρ) — оттенки одного синего тона, от светлого к тёмному
    (шкалы проверены валидатором палитры); L2 и L1 на одной панели — синий и оранжевый.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, PercentFormatter

COLOR_TRAIN = "#2a78d6"
COLOR_TEST = "#eb6834"
COLOR_L2 = "#2a78d6"
COLOR_L1 = "#eb6834"
COLOR_MARK = "#8a8984"
COLOR_CONTROL = "#8a8984"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#e4e3df"
COLOR_SURFACE = "#fcfcfb"
RAMP3 = ["#86b6ef", "#2a78d6", "#0d366b"]
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
KIND_COLORS = {"l2": COLOR_L2, "l1": COLOR_L1}
KIND_NAMES = {"l2": "L2", "l1": "L1"}
MAX_TICKS = 8


# ---------- оси и стиль ----------


def ramp(n: int) -> list[str]:
    """Упорядоченная шкала из n оттенков (n <= 5), от светлого к тёмному.

    Берутся подмножества проверенных шкал RAMP3 и RAMP5: у подмножества шаги между
    соседними оттенками только больше, поэтому проверка валидатором остаётся в силе.
    """
    if n == 1:
        return [RAMP3[1]]
    if n == 2:
        return [RAMP3[0], RAMP3[2]]
    if n == 3:
        return list(RAMP3)
    if n <= 5:
        return [RAMP5[i] for i in np.linspace(0, 4, n).round().astype(int)]
    raise ValueError("шкала поддерживает не больше 5 оттенков")


def log_ticks(low: float, high: float) -> list[float]:
    """Деления логарифмической оси: самый частый ряд, дающий от 3 до MAX_TICKS делений.

    На широком диапазоне получается 1-2-5 или 1-3, на узком (меньше порядка) — 1-1.5-2-3-...
    """
    options = ([1, 1.5, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 7], [1, 2, 5], [1, 3], [1])
    candidates = []
    for mantissas in options:
        ticks = [
            m * 10.0**e
            for e in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
            for m in mantissas
            if low <= m * 10.0**e <= high
        ]
        if 3 <= len(ticks) <= MAX_TICKS:
            return ticks
        candidates.append(ticks)
    fitting = [t for t in candidates if len(t) <= MAX_TICKS]
    return fitting[0] if fitting else candidates[-1]


def plain_number(x: float, _pos=None) -> str:
    """Число без степеней десяти: 0.2, 5, 1 000, 20 000."""
    if x >= 1000:
        return f"{round(x):,}".replace(",", " ")
    return f"{x:g}"


def sci_label(value: float) -> str:
    """Короткая запись малых величин: 1e-5, 3e-4."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def style_axes(ax) -> None:
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, which="major", color=COLOR_GRID, lw=0.8)
    ax.tick_params(colors=COLOR_TEXT_MUTED, which="both")
    for spine in ax.spines.values():
        spine.set_visible(False)


def setup_log_axis(axis, low: float, high: float) -> None:
    """Логарифмическая ось с подписями обычными числами (axis = ax.xaxis или ax.yaxis)."""
    axis.set_major_locator(FixedLocator(log_ticks(low, high)))
    axis.set_major_formatter(FuncFormatter(plain_number))
    axis.set_minor_formatter(NullFormatter())


def setup_epoch_axis(ax, steps: np.ndarray, label: str = "Эпоха (логарифмическая шкала)") -> None:
    positive = np.asarray(steps)[np.asarray(steps) > 0]
    low, high = float(positive[0]), float(positive[-1])
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    setup_log_axis(ax.xaxis, low, high)
    ax.set_xlabel(label, color=COLOR_TEXT_MUTED)


def setup_accuracy_axis(ax, label: str = "Точность, %") -> None:
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel(label, color=COLOR_TEXT_MUTED)


def setup_ce_axis(ax, values: np.ndarray, q: int) -> None:
    chance = math.log(q)
    finite = np.asarray(values)[np.isfinite(values)]
    low = min(float(finite.min()), chance) / 1.5
    high = max(float(finite.max()), chance) * 1.5
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    setup_log_axis(ax.yaxis, low, high)
    # Подпись линии — через легенду: так она не перекрывает кривые при любой их форме
    ax.axhline(chance, color=COLOR_MARK, lw=1, ls=":", label=f"случайное угадывание: ln {q} ≈ {chance:.2f}")
    ax.set_ylabel("Перекрёстная энтропия (логарифмическая шкала)", color=COLOR_TEXT_MUTED)


def mark_epochs(ax, marks: list[tuple[int | None, str, str]], fontsize: float = 9) -> None:
    """Вертикальные линии с подписями над графиком. marks: (эпоха, подпись, 'left' | 'right')."""
    for t, name, side in marks:
        if t is None or (isinstance(t, float) and math.isnan(t)):
            continue
        ax.axvline(t, color=COLOR_MARK, lw=1, ls="--")
        dx, ha = (-4, "right") if side == "left" else (4, "left")
        ax.annotate(
            f"{name} = {int(t)}", xy=(t, 1.0), xycoords=("data", "axes fraction"),
            xytext=(dx, 3), textcoords="offset points", ha=ha, va="bottom", fontsize=fontsize, color=COLOR_TEXT_MUTED,
        )


def save(fig, path: Path, title: str) -> None:
    fig.suptitle(title, color=COLOR_TEXT, fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=COLOR_SURFACE)
    plt.close(fig)


def legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, labelcolor=COLOR_TEXT, **{"fontsize": 9, **kwargs})


# ---------- кривые обучения (п. 2) ----------


def plot_replica(m: dict, r: int, t_mem, t_grok, q: int, title: str, path: Path) -> None:
    """Точность и перекрёстная энтропия одной реплики с отметками t_mem и t_grok."""
    mask = m["step"] > 0
    steps = m["step"][mask]
    fig, (ax_acc, ax_ce) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=COLOR_SURFACE)
    for ax, key_train, key_test in [(ax_acc, "train_acc", "test_acc"), (ax_ce, "train_loss", "test_loss")]:
        style_axes(ax)
        ax.plot(steps, m[key_train][mask, r], color=COLOR_TRAIN, lw=2, label="train")
        ax.plot(steps, m[key_test][mask, r], color=COLOR_TEST, lw=2, label="test")
        setup_epoch_axis(ax, m["step"])
        mark_epochs(ax, [(t_mem, "t_mem", "left"), (t_grok, "t_grok", "right")])
    setup_accuracy_axis(ax_acc)
    setup_ce_axis(ax_ce, np.concatenate([m["train_loss"][mask, r], m["test_loss"][mask, r]]), q)
    legend(ax_acc, loc="center left")
    legend(ax_ce, loc="lower left")
    save(fig, path, title)


def plot_lr_sweep(panels: list[tuple[float, dict, int, object]], q: int, path: Path) -> None:
    """Подбор η: по панели на η, train и test одной реплики, отметки t_mem и t_grok.

    panels — список (lr, metrics, номер реплики, строка детекции с t_mem, t_grok, outcome).
    """
    n_cols = 3
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.6 * n_rows), facecolor=COLOR_SURFACE, squeeze=False)
    outcome_names = {
        "grokking": "grokking", "partial": "частичное обобщение", "no_generalization": "нет обобщения",
        "simultaneous": "без задержки", "no_memorization": "нет запоминания",
    }
    for ax, (lr, m, r, det) in zip(axes.ravel(), panels):
        style_axes(ax)
        mask = m["step"] > 0
        ax.plot(m["step"][mask], m["train_acc"][mask, r], color=COLOR_TRAIN, lw=2, label="train")
        ax.plot(m["step"][mask], m["test_acc"][mask, r], color=COLOR_TEST, lw=2, label="test")
        setup_epoch_axis(ax, m["step"], label="Эпоха")
        setup_accuracy_axis(ax, label="")
        mark_epochs(ax, [(det.t_mem, "t_mem", "left"), (det.t_grok, "t_grok", "right")], fontsize=8)
        ax.set_title(f"η = {sci_label(lr)}: {outcome_names[det.outcome]}", color=COLOR_TEXT, fontsize=10, pad=16)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("Точность, %", color=COLOR_TEXT_MUTED)
    legend(axes[0, 0], loc="center left")
    save(fig, path, f"Подбор η: сложение по модулю {q}, L2, ρ = 1e-4, seed 0")


def plot_rho_sweep(m: dict, seed: int, title: str, path: Path) -> None:
    """Точность на train и test для всех ρ одного задания (один seed); ρ = 0 — серый пунктир."""
    mask = m["step"] > 0
    steps = m["step"][mask]
    replicas = [r for r in range(len(m["seed"])) if m["seed"][r] == seed]
    nonzero = sorted({float(m["rho"][r]) for r in replicas if m["rho"][r] > 0})
    colors = dict(zip(nonzero, ramp(len(nonzero))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=COLOR_SURFACE, sharey=True)
    for ax, key, name in [(axes[0], "train_acc", "train"), (axes[1], "test_acc", "test")]:
        style_axes(ax)
        for r in sorted(replicas, key=lambda i: m["rho"][i]):
            rho = float(m["rho"][r])
            if rho == 0:
                ax.plot(steps, m[key][mask, r], color=COLOR_CONTROL, lw=2, ls="--", label="ρ = 0 (контроль)")
            else:
                ax.plot(steps, m[key][mask, r], color=colors[rho], lw=2, label=f"ρ = {sci_label(rho)}")
        setup_epoch_axis(ax, m["step"])
        ax.set_title(f"Точность на {name}", color=COLOR_TEXT, fontsize=10)
    setup_accuracy_axis(axes[0])
    legend(axes[1], loc="upper left")
    save(fig, path, title)


# ---------- проверка критерия (п. 3) ----------


def plot_criterion_example(steps, train, test, t_star: float, estimates: dict[str, float | None], path: Path) -> None:
    """Синтетическая кривая с известным t*: слева целиком, справа — окрестность перехода с оценками критериев."""
    mask = steps > 0
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=COLOR_SURFACE,
                                           gridspec_kw={"width_ratios": [1.1, 1]})
    styles = [("--", COLOR_TRAIN), (":", COLOR_TEXT), ("-.", COLOR_TEST), ((0, (1, 2)), COLOR_MARK)]
    for ax in (ax_full, ax_zoom):
        style_axes(ax)
        ax.plot(steps[mask], train[mask], color=COLOR_TRAIN, lw=1.5, alpha=0.6, label="train (синтетика)")
        ax.plot(steps[mask], test[mask], color=COLOR_TEST, lw=1.5, alpha=0.8, label="test (синтетика)")
        ax.axvline(t_star, color=COLOR_TEXT, lw=2.5, alpha=0.35, label=f"истинный t* = {t_star:.0f}")
        setup_accuracy_axis(ax)
    for (name, t), (ls, color) in zip(estimates.items(), styles):
        if t is not None:
            ax_zoom.axvline(t, color=color, lw=1.6, ls=ls, label=f"{name}: {t:.0f}")
    setup_epoch_axis(ax_full, steps)
    low, high = t_star / 2.5, t_star * 2.5
    ax_zoom.set_xscale("log")
    ax_zoom.set_xlim(low, high)
    setup_log_axis(ax_zoom.xaxis, low, high)
    ax_zoom.set_xlabel("Эпоха (логарифмическая шкала), окрестность перехода", color=COLOR_TEXT_MUTED)
    ax_zoom.set_ylabel("")
    ax_full.set_title("Кривая целиком", color=COLOR_TEXT, fontsize=10)
    ax_zoom.set_title("Оценки критериев", color=COLOR_TEXT, fontsize=10)
    legend(ax_full, loc="center left", fontsize=8)
    legend(ax_zoom, loc="center left", fontsize=8)
    save(fig, path, "Синтетическая кривая с известным моментом перехода t* и оценки критериев")


def plot_criterion_errors(errors: dict[str, np.ndarray], path: Path) -> None:
    """Распределение ошибки |log10(t̂ / t*)| по критериям на синтетических кривых."""
    names = list(errors)
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.75 * len(names)), facecolor=COLOR_SURFACE)
    style_axes(ax)
    data = [np.asarray(errors[n], dtype=float) for n in names]
    box = ax.boxplot(data, vert=False, widths=0.5, showfliers=False, patch_artist=True)
    for patch in box["boxes"]:
        patch.set(facecolor="#cde2fb", edgecolor=COLOR_TRAIN)
    for key in ("whiskers", "caps", "medians"):
        for line in box[key]:
            line.set(color=COLOR_TRAIN if key != "medians" else COLOR_TEXT, lw=1.3)
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.set_xlabel("Ошибка |log₁₀(оценка / истинный t*)|: 0.04 ≈ ошибка 10%, 0.1 ≈ 26%", color=COLOR_TEXT_MUTED)
    ax.invert_yaxis()
    save(fig, path, "Точность критериев на синтетических кривых (ящик — 25–75%, усы — 5–95%)")


def plot_criterion_agreement(det: pd.DataFrame, path: Path) -> None:
    """Согласие основного критерия с альтернативными на реальных запусках сетки."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=COLOR_SURFACE)
    for ax, column, name in [(axes[0], "t_grok_thr50", "порог середины"), (axes[1], "t_grok_logderiv", "максимум производной")]:
        style_axes(ax)
        sub = det.dropna(subset=["t_grok", column])
        for kind in ("l2", "l1"):
            part = sub[sub.reg_kind == kind]
            ax.scatter(part["t_grok"], part[column], s=16, color=KIND_COLORS[kind], alpha=0.75, label=KIND_NAMES[kind],
                       edgecolor=COLOR_SURFACE, linewidth=0.5)
        low = float(min(sub["t_grok"].min(), sub[column].min())) / 1.2
        high = float(max(sub["t_grok"].max(), sub[column].max())) * 1.2
        ax.plot([low, high], [low, high], color=COLOR_MARK, lw=1, ls="--", label="полное совпадение")
        for axis_setter, axis in [(ax.set_xscale, ax.xaxis), (ax.set_yscale, ax.yaxis)]:
            axis_setter("log")
            setup_log_axis(axis, low, high)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("t_grok, основной критерий (ступенька)", color=COLOR_TEXT_MUTED)
        ax.set_ylabel(f"t_grok, {name}", color=COLOR_TEXT_MUTED)
        legend(ax, loc="upper left")
    save(fig, path, "Согласие критериев на реальных запусках сетки (каждая точка — реплика)")


# ---------- сетка (п. 4) ----------


def _errorbars(ax, x, med, q25, q75, color, label, marker="o", ls="-"):
    ax.errorbar(x, med, yerr=[med - q25, q75 - med], color=color, lw=1.8, marker=marker, ms=6, capsize=3,
                ls=ls, label=label)


def plot_tgrok_vs(summary: pd.DataFrame, x: str, by: str, path: Path, title: str) -> None:
    """Медиана и межквартильный размах t_grok: ось x — η или ρ, линии — значения другого фактора.

    По панели на тип регуляризации; ячейки без наблюдённых переходов не рисуются.
    """
    kinds = [k for k in ("l2", "l1") if k in set(summary.reg_kind)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.2 * len(kinds), 4.6), facecolor=COLOR_SURFACE, sharey=True,
                             squeeze=False)
    # Ячейки, где переход наступил меньше чем у половины seed (медиана «> T»), не рисуются
    sub_all = summary[(summary.rho > 0) & np.isfinite(summary.t_grok_median)].copy()
    finite_q75 = sub_all.t_grok_q75[np.isfinite(sub_all.t_grok_q75)]
    y_low, y_high = sub_all.t_grok_q25.min() / 1.4, max(finite_q75.max(), sub_all.t_grok_median.max()) * 1.4
    # Бесконечный верхний квартиль (переход не у всех seed) обрезается краем графика
    sub_all["t_grok_q75"] = np.where(np.isfinite(sub_all.t_grok_q75), sub_all.t_grok_q75, y_high)
    x_label = {"lr": "Скорость обучения η", "rho": "Коэффициент регуляризации ρ"}[x]
    by_label = {"lr": "η", "rho": "ρ"}[by]
    for ax, kind in zip(axes[0], kinds):
        style_axes(ax)
        part = sub_all[sub_all.reg_kind == kind]
        levels = sorted(part[by].unique())
        for level, color in zip(levels, ramp(len(levels))):
            line = part[part[by] == level].sort_values(x)
            _errorbars(ax, line[x].to_numpy(), line.t_grok_median.to_numpy(), line.t_grok_q25.to_numpy(),
                       line.t_grok_q75.to_numpy(), color, f"{by_label} = {sci_label(level)}")
        x_low, x_high = part[x].min() / 1.5, part[x].max() * 1.5
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)
        ax.xaxis.set_major_locator(FixedLocator(sorted(part[x].unique())))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: sci_label(v)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        setup_log_axis(ax.yaxis, y_low, y_high)
        ax.set_xlabel(x_label, color=COLOR_TEXT_MUTED)
        ax.set_title(KIND_NAMES[kind], color=COLOR_TEXT, fontsize=10)
        legend(ax, loc="upper right" if x == "lr" else "lower left", fontsize=8)
    axes[0][0].set_ylabel("t_grok, эпох (медиана и 25–75%)", color=COLOR_TEXT_MUTED)
    save(fig, path, title)


def plot_end_accuracy(summary: pd.DataFrame, path: Path) -> None:
    """Итоговая test_acc (среднее за последние 5% эпох) по ρ: L2 и L1 на одной панели, панель на η."""
    lrs = sorted(summary.lr.unique())
    fig, axes = plt.subplots(1, len(lrs), figsize=(4.4 * len(lrs), 4.2), facecolor=COLOR_SURFACE, sharey=True,
                             squeeze=False)
    sub_all = summary[summary.rho > 0]
    x_low, x_high = sub_all.rho.min() / 1.6, sub_all.rho.max() * 1.6
    for ax, lr in zip(axes[0], lrs):
        style_axes(ax)
        for kind in ("l2", "l1"):
            part = sub_all[(sub_all.lr == lr) & (sub_all.reg_kind == kind)].sort_values("rho")
            if part.empty:
                continue
            _errorbars(ax, part.rho.to_numpy(), part.end_acc_median.to_numpy(), part.end_acc_q25.to_numpy(),
                       part.end_acc_q75.to_numpy(), KIND_COLORS[kind], KIND_NAMES[kind])
        ax.set_xscale("log")
        ax.set_xlim(x_low, x_high)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: sci_label(v)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        setup_accuracy_axis(ax, label="")
        ax.set_xlabel("Коэффициент регуляризации ρ", color=COLOR_TEXT_MUTED)
        ax.set_title(f"η = {sci_label(lr)}", color=COLOR_TEXT, fontsize=10)
    axes[0][0].set_ylabel("Итоговая точность на test, %", color=COLOR_TEXT_MUTED)
    legend(axes[0][0], loc="lower left")
    save(fig, path, "Итоговая точность на test (медиана и 25–75% по seed)")


def plot_norm_dynamics(rows: list[tuple[str, dict, int, object]], path: Path) -> None:
    """Точность, L2-норма весов и доля почти нулевых весов во времени для выбранных реплик.

    rows — список (подпись строки, metrics, номер реплики, строка детекции).
    """
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 3.6 * len(rows)), facecolor=COLOR_SURFACE, squeeze=False)
    for (label, m, r, det), row_axes in zip(rows, axes):
        mask = m["step"] > 0
        steps = m["step"][mask]
        panels = [
            (row_axes[0], None, "Точность, %"),
            (row_axes[1], "l2_norm", "L2-норма весов √Σw²"),
            (row_axes[2], "frac_small", "Доля весов с |w| < 0.001, %"),
        ]
        for ax, key, ylabel in panels:
            style_axes(ax)
            if key is None:
                ax.plot(steps, m["train_acc"][mask, r], color=COLOR_TRAIN, lw=2, label="train")
                ax.plot(steps, m["test_acc"][mask, r], color=COLOR_TEST, lw=2, label="test")
                setup_accuracy_axis(ax, label=ylabel)
                legend(ax, loc="center left")
            else:
                ax.plot(steps, m[key][mask, r], color=COLOR_TEXT_MUTED, lw=2)
                ax.set_ylabel(ylabel, color=COLOR_TEXT_MUTED)
                if key == "frac_small":
                    # при долях в единицы процентов нужен знак после запятой, иначе подписи повторяются
                    decimals = 1 if float(np.max(m[key][mask, r])) < 0.1 else 0
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=decimals))
            setup_epoch_axis(ax, m["step"], label="Эпоха")
            mark_epochs(ax, [(det.t_mem, "t_mem", "left"), (det.t_grok, "t_grok", "right")], fontsize=8)
        row_axes[0].set_title(label, color=COLOR_TEXT, fontsize=10, loc="left", pad=16)
    save(fig, path, "Что происходит с весами во время перехода")


# ---------- механизм: Фурье-структура первого слоя ----------


def plot_fourier(runs: list[dict], example: dict, path: Path) -> None:
    """Механизм перехода: периодичность нейронов и число живых нейронов по эпохам, пример весов.

    runs — словари: label, color, ls, steps, periodicity, alive, t_grok.
    example — спектры нескольких нейронов по входу x1 в конце обучения: label, spectra, names, init_spectrum.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=COLOR_SURFACE,
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax_per, ax_alive, ax_ex = axes
    for ax in axes:
        style_axes(ax)
    all_steps = np.unique(np.concatenate([run["steps"] for run in runs]))
    for run in runs:
        steps = np.asarray(run["steps"])
        mask = steps > 0
        for ax, key in ((ax_per, "periodicity"), (ax_alive, "alive")):
            values = np.asarray(run[key], dtype=float)
            ax.plot(steps[mask], values[mask], color=run["color"], ls=run["ls"], lw=2, marker="o", ms=3.5,
                    label=run["label"])
            if run["t_grok"] is not None:
                y = float(np.interp(np.log(run["t_grok"]), np.log(steps[mask]), values[mask]))
                ax.plot([run["t_grok"]], [y], marker="D", ms=8, color=run["color"], mec=COLOR_SURFACE, mew=1.2,
                        ls="none")
    for ax in (ax_per, ax_alive):
        setup_epoch_axis(ax, all_steps, label="Эпоха")
    ax_per.set_ylim(0, 1.0)
    ax_per.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_per.set_ylabel("Доля мощности нейрона в его главной частоте", color=COLOR_TEXT_MUTED)
    ax_per.set_title("Периодичность нейронов (ромб — t_grok)", color=COLOR_TEXT, fontsize=10)
    legend(ax_per, loc="upper left", fontsize=8)
    ax_alive.set_ylim(0, 270)
    ax_alive.set_ylabel("Живых нейронов из 256", color=COLOR_TEXT_MUTED)
    ax_alive.set_title("Сколько нейронов участвует в решении", color=COLOR_TEXT, fontsize=10)

    k = np.arange(1, len(example["init_spectrum"]) + 1)
    ax_ex.plot(k, example["init_spectrum"], color=COLOR_MARK, lw=1.2, label="случайный нейрон при инициализации")
    for spectrum, color, name in zip(example["spectra"], RAMP3[::-1], example["names"]):
        ax_ex.plot(k, spectrum, color=color, lw=1.8, label=name)
    ax_ex.set_xlim(1, k[-1])
    ax_ex.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_ex.set_xlabel("Частота k по кругу остатков (1…48)", color=COLOR_TEXT_MUTED)
    ax_ex.set_ylabel("Доля мощности нейрона", color=COLOR_TEXT_MUTED)
    ax_ex.set_title(example["label"], color=COLOR_TEXT, fontsize=10)
    legend(ax_ex, loc="upper right", fontsize=8)
    save(fig, path, "Механизм перехода: нейроны первого слоя становятся синусоидами по входу")
