"""
Общая отрисовка графиков пилота: единый стиль и читаемые оси.

Соглашения по осям:
  * эпохи — логарифмическая шкала, подписи обычными числами (50, 100, 1 000, 5 000),
    эпоха 0 не рисуется: на логарифмической оси её нельзя показать;
  * точность — в процентах от 0 до 100;
  * перекрёстная энтропия — логарифмическая шкала с подписями обычными числами
    (0.1, 0.5, 1, 5, 10) и пунктиром на уровне случайного угадывания ln q.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, PercentFormatter

# Цвета: первые два слота категориальной палитры, текст и сетка — нейтральные
COLOR_TRAIN = "#2a78d6"
COLOR_TEST = "#eb6834"
COLOR_MARK = "#8a8984"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#e4e3df"
COLOR_SURFACE = "#fcfcfb"
# Упорядоченная шкала для rho: один тон, от светлого к тёмному (проверена валидатором палитры)
RHO_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
COLOR_CONTROL = "#8a8984"

MAX_TICKS = 8


def log_ticks(low: float, high: float) -> list[float]:
    """Деления логарифмической оси из ряда 1-2-5 (или 1-3, или 1), не больше MAX_TICKS."""
    for mantissas in ([1, 2, 5], [1, 3], [1]):
        ticks = [
            m * 10.0 ** e
            for e in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
            for m in mantissas
            if low <= m * 10.0 ** e <= high
        ]
        if len(ticks) <= MAX_TICKS:
            return ticks
    return ticks


def plain_number(x: float, _pos=None) -> str:
    """Число без степеней десяти: 0.2, 5, 1 000, 20 000."""
    if x >= 1000:
        return f"{int(round(x)):,}".replace(",", " ")
    return f"{x:g}"


def rho_label(rho: float) -> str:
    """Короткая запись rho: 1e-5, 3e-4."""
    mantissa, exponent = f"{rho:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def style_axes(ax) -> None:
    """Фон, сетка, цвет подписей; без рамки."""
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, which="major", color=COLOR_GRID, lw=0.8)
    ax.tick_params(colors=COLOR_TEXT_MUTED, which="both")
    for spine in ax.spines.values():
        spine.set_visible(False)


def positive_steps(steps: np.ndarray) -> np.ndarray:
    """Маска эпох > 0: эпоху 0 на логарифмической оси показать нельзя."""
    return np.asarray(steps) > 0


def setup_epoch_axis(ax, steps: np.ndarray) -> None:
    """Ось эпох: логарифмическая, от первой записанной эпохи после 0 до последней."""
    positive = np.asarray(steps)[positive_steps(steps)]
    low, high = float(positive[0]), float(positive[-1])
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(FixedLocator(log_ticks(low, high)))
    ax.xaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Эпоха (логарифмическая шкала)", color=COLOR_TEXT_MUTED)


def setup_accuracy_axis(ax, label: str = "Точность, %") -> None:
    """Ось точности: 0–100%."""
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel(label, color=COLOR_TEXT_MUTED)


def setup_ce_axis(ax, values: np.ndarray, q: int) -> None:
    """Ось перекрёстной энтропии: логарифмическая, с уровнем случайного угадывания ln q."""
    chance = math.log(q)
    finite = np.asarray(values)[np.isfinite(values)]
    low = min(float(finite.min()), chance) / 1.5
    high = max(float(finite.max()), chance) * 1.5
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(FixedLocator(log_ticks(low, high)))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Подпись линии — через легенду: так она не перекрывает кривые при любой их форме
    ax.axhline(chance, color=COLOR_MARK, lw=1, ls=":", label=f"случайное угадывание: ln {q} ≈ {chance:.2f}")
    ax.set_ylabel("Перекрёстная энтропия (логарифмическая шкала)", color=COLOR_TEXT_MUTED)


def mark_epochs(ax, marks: list[tuple[int | None, str, str]]) -> None:
    """Вертикальные линии с подписями над графиком. marks: (эпоха, подпись, 'left' | 'right')."""
    for t, name, side in marks:
        if t is None:
            continue
        ax.axvline(t, color=COLOR_MARK, lw=1, ls="--")
        dx, ha = (-4, "right") if side == "left" else (4, "left")
        ax.annotate(
            f"{name} = {t}", xy=(t, 1.0), xycoords=("data", "axes fraction"),
            xytext=(dx, 3), textcoords="offset points", ha=ha, va="bottom", fontsize=9, color=COLOR_TEXT_MUTED,
        )


def save(fig, path: Path, title: str) -> None:
    """Заголовок, компоновка, сохранение."""
    fig.suptitle(title, color=COLOR_TEXT, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=COLOR_SURFACE)
    plt.close(fig)


def plot_replica(m, r: int, det: dict, q: int, title: str, path: Path) -> None:
    """Точность и перекрёстная энтропия одной реплики с отметками t_mem и t_grok."""
    mask = positive_steps(m["step"])
    steps = m["step"][mask]
    fig, (ax_acc, ax_ce) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=COLOR_SURFACE)
    for ax, key_train, key_test in [(ax_acc, "train_acc", "test_acc"), (ax_ce, "train_loss", "test_loss")]:
        style_axes(ax)
        ax.plot(steps, m[key_train][mask, r], color=COLOR_TRAIN, lw=2, label="train")
        ax.plot(steps, m[key_test][mask, r], color=COLOR_TEST, lw=2, label="test")
        setup_epoch_axis(ax, m["step"])
        mark_epochs(ax, [(det["t_mem"], "t_mem", "left"), (det["t_grok"], "t_grok", "right")])
    setup_accuracy_axis(ax_acc)
    ce_values = np.concatenate([m["train_loss"][mask, r], m["test_loss"][mask, r]])
    setup_ce_axis(ax_ce, ce_values, q)
    ax_acc.legend(frameon=False, loc="center left", labelcolor=COLOR_TEXT, fontsize=9)
    # Левый нижний угол свободен: в начале обучения обе кривые около ln q
    ax_ce.legend(frameon=False, loc="lower left", labelcolor=COLOR_TEXT, fontsize=9)
    save(fig, path, title)


def plot_rho_sweep(m, seed: int, title: str, path: Path) -> None:
    """Точность на train и test для всех rho одного запуска (один seed).

    rho = 0 рисуется серым пунктиром как контроль, ненулевые rho — шкалой одного тона:
    чем больше rho, тем темнее линия. Поддерживается до len(RHO_RAMP) ненулевых rho.
    """
    mask = positive_steps(m["step"])
    steps = m["step"][mask]
    replicas = [r for r in range(len(m["seed"])) if m["seed"][r] == seed]
    nonzero = sorted({float(m["rho"][r]) for r in replicas if m["rho"][r] > 0})
    if len(nonzero) > len(RHO_RAMP):
        raise ValueError(f"не больше {len(RHO_RAMP)} ненулевых rho на графике, получено {len(nonzero)}")
    # Самые тёмные шаги шкалы — самым большим rho
    colors = dict(zip(nonzero, RHO_RAMP[len(RHO_RAMP) - len(nonzero):]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=COLOR_SURFACE, sharey=True)
    for ax, key, name in [(axes[0], "train_acc", "train"), (axes[1], "test_acc", "test")]:
        style_axes(ax)
        for r in sorted(replicas, key=lambda i: m["rho"][i]):
            rho = float(m["rho"][r])
            if rho == 0:
                ax.plot(steps, m[key][mask, r], color=COLOR_CONTROL, lw=2, ls="--", label="ρ = 0 (контроль)")
            else:
                ax.plot(steps, m[key][mask, r], color=colors[rho], lw=2, label=f"ρ = {rho_label(rho)}")
        setup_epoch_axis(ax, m["step"])
        ax.set_title(f"Точность на {name}", color=COLOR_TEXT, fontsize=10)
    setup_accuracy_axis(axes[0])
    axes[1].set_ylabel("")
    axes[1].legend(frameon=False, loc="upper left", labelcolor=COLOR_TEXT, fontsize=9)
    save(fig, path, title)


def plot_run_curves(arrays: dict, replica_rho: list, replica_seed: list, q: int, title: str, path: Path) -> None:
    """Рабочий график запуска pilot_grokking.py: все реплики, пунктир — train, сплошная — test."""
    mask = positive_steps(arrays["step"])
    steps = arrays["step"][mask]
    fig, (ax_acc, ax_ce) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=COLOR_SURFACE)
    for ax in (ax_acc, ax_ce):
        style_axes(ax)
    for r in range(len(replica_seed)):
        label = f"ρ = {rho_label(replica_rho[r]) if replica_rho[r] > 0 else '0'}, seed = {replica_seed[r]}"
        line = ax_acc.plot(steps, arrays["train_acc"][mask, r], lw=1, ls="--")[0]
        ax_acc.plot(steps, arrays["test_acc"][mask, r], lw=1.5, color=line.get_color(), label=label)
        ax_ce.plot(steps, arrays["train_loss"][mask, r], lw=1, ls="--", color=line.get_color())
        ax_ce.plot(steps, arrays["test_loss"][mask, r], lw=1.5, color=line.get_color())
    for ax in (ax_acc, ax_ce):
        setup_epoch_axis(ax, arrays["step"])
    setup_accuracy_axis(ax_acc)
    ce_values = np.concatenate([arrays["train_loss"][mask].ravel(), arrays["test_loss"][mask].ravel()])
    setup_ce_axis(ax_ce, ce_values, q)
    ax_acc.set_title("Точность (пунктир — train, сплошная — test)", color=COLOR_TEXT, fontsize=10)
    ax_ce.set_title("Перекрёстная энтропия (пунктир — train, сплошная — test)", color=COLOR_TEXT, fontsize=10)
    ax_acc.legend(fontsize=7, loc="upper left", frameon=False, labelcolor=COLOR_TEXT)
    ax_ce.legend(fontsize=8, loc="best", frameon=False, labelcolor=COLOR_TEXT)
    save(fig, path, title)
