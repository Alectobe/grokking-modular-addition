"""
Пилот критерия момента перехода (docs/EXPERIMENTS.md §3) на кривых из pilot_grokking.py.

Для каждой реплики считает:
  t_mem      — первая эпоха с train_acc >= 0.99;
  t_grok     — основной критерий: лучшая аппроксимация test_acc ступенькой с одним
               скачком на отрезке после t_mem (минимум суммы квадратов отклонений);
  t_thr50    — альтернативный критерий: первая эпоха после t_mem, где test_acc
               пересекает середину между test_acc(t_mem) и итоговым значением;
  outcome    — исход по таблице из §3.1 (grokking, partial, no_generalization,
               simultaneous, no_memorization).

С флагом --plot дополнительно рисует кривые одной реплики (выбор по --rho и --seed
в первом из переданных запусков) с отмеченными t_mem и t_grok, с флагом --plot-sweep —
точность на train и test для всех rho первого запуска.

Пример запуска:
    python pilot/pilot_changepoint.py pilot_runs/l2_lr1e-3 pilot_runs/l1_lr1e-3
    python pilot/pilot_changepoint.py pilot_runs/l2_lr1e-3 --plot docs/img/pilot_grokking.png \
        --rho 1e-4 --seed 0
    python pilot/pilot_changepoint.py pilot_runs/l2_lr1e-3 --plot-sweep docs/img/pilot_l2_rho.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

MEM_THRESHOLD = 0.99
GEN_THRESHOLD = 0.95
SIMULTANEOUS_THRESHOLD = 0.5
MIN_JUMP = 0.3
TAIL_FRACTION = 0.05


def changepoint_l2(values: np.ndarray) -> tuple[int, float]:
    """Точка разрыва k (1 <= k < n), минимизирующая SSE ступеньки с одним скачком.

    Левый сегмент values[:k], правый values[k:]. Считается за O(n) через префиксные
    суммы. При равенстве SSE возвращается наименьший k.
    """
    n = len(values)
    if n < 2:
        raise ValueError("нужно хотя бы 2 точки")
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    csum2 = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64) ** 2)])
    k = np.arange(1, n)
    left_n, right_n = k, n - k
    left_sum, right_sum = csum[k], csum[n] - csum[k]
    left_sum2, right_sum2 = csum2[k], csum2[n] - csum2[k]
    # SSE сегмента = сумма квадратов - (сумма)^2 / длина
    sse = (left_sum2 - left_sum ** 2 / left_n) + (right_sum2 - right_sum ** 2 / right_n)
    best = int(np.argmin(sse))
    return int(k[best]), float(sse[best])


def detect(steps: np.ndarray, train_acc: np.ndarray, test_acc: np.ndarray) -> dict:
    """Момент запоминания, момент перехода и исход для одной реплики."""
    mem_hits = np.nonzero(train_acc >= MEM_THRESHOLD)[0]
    if len(mem_hits) == 0:
        return {"outcome": "no_memorization", "t_mem": None, "t_grok": None, "t_thr50": None, "jump": None}
    i_mem = int(mem_hits[0])
    t_mem = int(steps[i_mem])
    tail = test_acc[-max(1, int(round(TAIL_FRACTION * len(test_acc)))):]

    if test_acc[i_mem] >= SIMULTANEOUS_THRESHOLD:
        return {"outcome": "simultaneous", "t_mem": t_mem, "t_grok": None, "t_thr50": None, "jump": None}

    segment = test_acc[i_mem:]
    if len(segment) < 2:
        return {"outcome": "no_generalization", "t_mem": t_mem, "t_grok": None, "t_thr50": None, "jump": None}
    k, _ = changepoint_l2(segment)
    jump = float(segment[k:].mean() - segment[:k].mean())
    if jump < MIN_JUMP:
        # Скачка нет: переход не наступил до конца обучения (цензурированное наблюдение)
        return {"outcome": "no_generalization", "t_mem": t_mem, "t_grok": None, "t_thr50": None, "jump": jump}
    t_grok = int(steps[i_mem + k])

    # Альтернативный критерий: пересечение середины между уровнем в t_mem и итоговым
    midpoint = 0.5 * (test_acc[i_mem] + tail.mean())
    cross = np.nonzero(segment >= midpoint)[0]
    t_thr50 = int(steps[i_mem + cross[0]]) if len(cross) else None

    # Скачок есть; полное обобщение — если в конце обучения test_acc >= GEN_THRESHOLD
    outcome = "grokking" if tail.mean() >= GEN_THRESHOLD else "partial"
    return {"outcome": outcome, "t_mem": t_mem, "t_grok": t_grok, "t_thr50": t_thr50, "jump": jump}


def plot_replica(m, r: int, det: dict, title: str, path: Path) -> None:
    """Точность и лосс одной реплики по эпохам (лог-шкала) с отметками t_mem и t_grok."""
    steps = np.maximum(m["step"], 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=COLOR_SURFACE)
    panels = [
        (axes[0], "train_acc", "test_acc", "Точность", "linear"),
        (axes[1], "train_loss", "test_loss", "Перекрёстная энтропия", "log"),
    ]
    for ax, key_train, key_test, ylabel, yscale in panels:
        ax.set_facecolor(COLOR_SURFACE)
        ax.plot(steps, m[key_train][:, r], color=COLOR_TRAIN, lw=2, label="train")
        ax.plot(steps, m[key_test][:, r], color=COLOR_TEST, lw=2, label="test")
        # Подписи над областью графика: t_mem слева от своей линии, t_grok — справа
        for t, name, dx, ha in [(det["t_mem"], "t_mem", -4, "right"), (det["t_grok"], "t_grok", 4, "left")]:
            if t is not None:
                ax.axvline(t, color=COLOR_MARK, lw=1, ls="--")
                ax.annotate(
                    f"{name} = {t}", xy=(t, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(dx, 3), textcoords="offset points", ha=ha, va="bottom",
                    fontsize=9, color=COLOR_TEXT_MUTED,
                )
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlabel("Эпоха (лог. шкала)", color=COLOR_TEXT_MUTED)
        ax.set_ylabel(ylabel, color=COLOR_TEXT_MUTED)
        ax.grid(True, color=COLOR_GRID, lw=0.8)
        ax.tick_params(colors=COLOR_TEXT_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(frameon=False, loc="center left", labelcolor=COLOR_TEXT)
    fig.suptitle(title, color=COLOR_TEXT, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=COLOR_SURFACE)
    plt.close(fig)


def plot_rho_sweep(m, seed: int, title: str, path: Path) -> None:
    """Точность на train и test для всех rho одного запуска (один seed).

    rho = 0 рисуется серым пунктиром как контроль, ненулевые rho — шкалой одного тона:
    чем больше rho, тем темнее линия. Поддерживается до len(RHO_RAMP) ненулевых rho.
    """
    steps = np.maximum(m["step"], 1)
    replicas = [r for r in range(len(m["seed"])) if m["seed"][r] == seed]
    nonzero = sorted({float(m["rho"][r]) for r in replicas if m["rho"][r] > 0})
    if len(nonzero) > len(RHO_RAMP):
        raise ValueError(f"не больше {len(RHO_RAMP)} ненулевых rho на графике, получено {len(nonzero)}")
    # Самые тёмные шаги шкалы — самым большим rho
    colors = dict(zip(nonzero, RHO_RAMP[len(RHO_RAMP) - len(nonzero):]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=COLOR_SURFACE, sharey=True)
    for ax, key, name in [(axes[0], "train_acc", "train"), (axes[1], "test_acc", "test")]:
        ax.set_facecolor(COLOR_SURFACE)
        for r in sorted(replicas, key=lambda i: m["rho"][i]):
            rho = float(m["rho"][r])
            if rho == 0:
                ax.plot(steps, m[key][:, r], color=COLOR_CONTROL, lw=2, ls="--", label="ρ = 0 (контроль)")
            else:
                ax.plot(steps, m[key][:, r], color=colors[rho], lw=2, label=f"ρ = {rho:g}")
        ax.set_xscale("log")
        ax.set_title(f"Точность на {name}", color=COLOR_TEXT, fontsize=10)
        ax.set_xlabel("Эпоха (лог. шкала)", color=COLOR_TEXT_MUTED)
        ax.grid(True, color=COLOR_GRID, lw=0.8)
        ax.tick_params(colors=COLOR_TEXT_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Точность", color=COLOR_TEXT_MUTED)
    axes[1].legend(frameon=False, loc="upper left", labelcolor=COLOR_TEXT, fontsize=9)
    fig.suptitle(title, color=COLOR_TEXT, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=COLOR_SURFACE)
    plt.close(fig)


def run_title(run_dir: Path, extra: str) -> str:
    """Заголовок графика из параметров запуска, сохранённых в summary.json."""
    a = json.loads((run_dir / "summary.json").read_text())["args"]
    return f"Сложение по модулю {a['q']}: {a['reg'].upper()}, {a['optimizer'].capitalize()}, η = {a['lr']:g}{extra}"


def main():
    parser = argparse.ArgumentParser(description="Критерий момента перехода на кривых пилота")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--plot", type=Path, default=None, help="куда сохранить график одной реплики")
    parser.add_argument("--rho", type=float, default=None, help="rho реплики для графика")
    parser.add_argument("--seed", type=int, default=0, help="seed реплики для графика")
    parser.add_argument("--plot-sweep", type=Path, default=None,
                        help="куда сохранить график всех rho первого запуска (seed из --seed)")
    args = parser.parse_args()

    if args.plot is not None:
        m = np.load(args.run_dirs[0] / "metrics.npz")
        matches = np.nonzero(np.isclose(m["rho"], args.rho) & (m["seed"] == args.seed))[0]
        if len(matches) == 0:
            raise SystemExit(f"в {args.run_dirs[0]} нет реплики rho={args.rho}, seed={args.seed}")
        r = int(matches[0])
        det = detect(m["step"], m["train_acc"][:, r], m["test_acc"][:, r])
        title = run_title(args.run_dirs[0], f", ρ = {args.rho:g}, seed = {args.seed}")
        plot_replica(m, r, det, title, args.plot)
        print(f"график сохранён: {args.plot}")

    if args.plot_sweep is not None:
        m = np.load(args.run_dirs[0] / "metrics.npz")
        title = run_title(args.run_dirs[0], f", seed = {args.seed}")
        plot_rho_sweep(m, args.seed, title, args.plot_sweep)
        print(f"график сохранён: {args.plot_sweep}")

    header = f"{'запуск':<16} {'rho':>8} {'seed':>4}  {'исход':<18} {'t_mem':>6} {'t_grok':>6} {'t_thr50':>7} {'скачок':>6}"
    print(header)
    print("-" * len(header))
    for run_dir in args.run_dirs:
        m = np.load(run_dir / "metrics.npz")
        for r in range(m["train_acc"].shape[1]):
            d = detect(m["step"], m["train_acc"][:, r], m["test_acc"][:, r])
            fmt = lambda v: "—" if v is None else str(v)
            jump = "—" if d["jump"] is None else f"{d['jump']:.2f}"
            print(
                f"{run_dir.name:<16} {m['rho'][r]:>8g} {m['seed'][r]:>4d}  {d['outcome']:<18} "
                f"{fmt(d['t_mem']):>6} {fmt(d['t_grok']):>6} {fmt(d['t_thr50']):>7} {jump:>6}"
            )


if __name__ == "__main__":
    main()
