"""
Критерий момента перехода от запоминания к обобщению (docs/EXPERIMENTS.md §3).

Основной критерий — задача оптимизации: на отрезке после запоминания кривая test_acc
приближается ступенькой с одним скачком по методу наименьших квадратов, момент скачка
и есть t_grok. Альтернативные критерии (порог середины, максимум производной по log t,
середина подогнанной сигмоиды) нужны для проверки основного.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


@dataclass(frozen=True)
class DetectionParams:
    mem_threshold: float = 0.99  # train_acc, с которой выборка считается запомненной
    gen_threshold: float = 0.95  # test_acc полного обобщения
    min_jump: float = 0.3  # минимальный скачок test_acc, чтобы считать переход состоявшимся
    simultaneous_threshold: float = 0.5  # test_acc в момент запоминания, выше которой задержки нет
    tail_fraction: float = 0.05  # доля последних точек для итогового уровня test_acc
    smooth_window: int = 5  # окно сглаживания (в точках) для критерия по производной


@dataclass(frozen=True)
class Detection:
    outcome: str  # grokking | partial | simultaneous | no_generalization | no_memorization
    t_mem: int | None
    t_grok: int | None
    censored: bool  # скачка не было до конца обучения: t_grok > T
    jump: float | None
    t_grok_thr50: int | None
    t_grok_logderiv: int | None
    t_gen95: int | None
    censored_gen95: bool  # test_acc >= gen_threshold не достигнута к концу обучения
    end_test_acc: float
    final_train_acc: float
    final_test_acc: float

    def as_row(self) -> dict:
        return asdict(self)


def changepoint_l2(values: np.ndarray) -> tuple[int, float]:
    """Точка разрыва k (1 <= k < n), минимизирующая сумму квадратов отклонений ступеньки.

    Левый сегмент values[:k], правый values[k:]. SSE сегмента = Σa² − (Σa)²/длина, поэтому
    через префиксные суммы задача решается точно за O(n). При равенстве берётся наименьший k.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        raise ValueError("нужно хотя бы 2 точки")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    csum2 = np.concatenate([[0.0], np.cumsum(values**2)])
    k = np.arange(1, n)
    left_sum, right_sum = csum[k], csum[n] - csum[k]
    left_sum2, right_sum2 = csum2[k], csum2[n] - csum2[k]
    sse = (left_sum2 - left_sum**2 / k) + (right_sum2 - right_sum**2 / (n - k))
    best = int(np.argmin(sse))
    return int(k[best]), float(sse[best])


def threshold_crossing(steps: np.ndarray, test_acc: np.ndarray, start_idx: int, level: float) -> int | None:
    """Первая эпоха с индексом >= start_idx, где test_acc >= level."""
    hits = np.nonzero(test_acc[start_idx:] >= level)[0]
    return int(steps[start_idx + hits[0]]) if len(hits) else None


def max_log_derivative(steps: np.ndarray, test_acc: np.ndarray, start_idx: int, window: int) -> int | None:
    """Эпоха наибольшей скорости роста сглаженной test_acc по log t (среднее геометрическое соседних точек)."""
    t = steps[start_idx:].astype(np.float64)
    a = test_acc[start_idx:].astype(np.float64)
    mask = t > 0
    t, a = t[mask], a[mask]
    if len(t) < window + 2:
        return None
    kernel = np.ones(window) / window
    smooth = np.convolve(a, kernel, mode="valid")
    t_valid = t[window // 2 : window // 2 + len(smooth)]
    slope = np.diff(smooth) / np.diff(np.log(t_valid))
    i = int(np.argmax(slope))
    return round(math.sqrt(t_valid[i] * t_valid[i + 1]))


def _sigmoid(u, low, high, u0, width):
    return low + (high - low) / (1.0 + np.exp(-(u - u0) / width))


def sigmoid_midpoint(steps: np.ndarray, test_acc: np.ndarray, start_idx: int, t_init: int) -> int | None:
    """Середина сигмоиды по log t, подогнанной МНК к test_acc после запоминания; None, если подгонка не сошлась."""
    t = steps[start_idx:].astype(np.float64)
    a = test_acc[start_idx:].astype(np.float64)
    mask = t > 0
    u, a = np.log(t[mask]), a[mask]
    p0 = [a[: max(1, len(a) // 20)].mean(), a[-max(1, len(a) // 20) :].mean(), math.log(t_init), 0.2]
    bounds = ([-0.1, 0.0, u.min(), 1e-3], [1.0, 1.1, u.max(), 5.0])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(_sigmoid, u, a, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError):
        return None
    return round(math.exp(params[2]))


def stable_crossing(steps: np.ndarray, values: np.ndarray, level: float) -> int | None:
    """Первая эпоха, начиная с которой values >= level до конца записи; None, если в конце ниже level."""
    below = np.nonzero(values < level)[0]
    if len(below) == 0:
        return int(steps[0])
    last_below = int(below[-1])
    return int(steps[last_below + 1]) if last_below + 1 < len(values) else None


def detect(
    steps: np.ndarray, train_acc: np.ndarray, test_acc: np.ndarray, params: DetectionParams | None = None
) -> Detection:
    """Момент запоминания, момент перехода и исход для одной реплики (docs/EXPERIMENTS.md §3.1)."""
    p = params or DetectionParams()
    steps = np.asarray(steps)
    train_acc = np.asarray(train_acc, dtype=np.float64)
    test_acc = np.asarray(test_acc, dtype=np.float64)
    n_tail = max(1, round(p.tail_fraction * len(test_acc)))
    end_test = float(test_acc[-n_tail:].mean())
    t_gen95 = stable_crossing(steps, test_acc, p.gen_threshold)
    common = {
        "t_gen95": t_gen95,
        "censored_gen95": t_gen95 is None,
        "end_test_acc": end_test,
        "final_train_acc": float(train_acc[-1]),
        "final_test_acc": float(test_acc[-1]),
    }
    empty = {"t_grok": None, "jump": None, "t_grok_thr50": None, "t_grok_logderiv": None}

    mem_hits = np.nonzero(train_acc >= p.mem_threshold)[0]
    if len(mem_hits) == 0:
        return Detection(outcome="no_memorization", t_mem=None, censored=False, **empty, **common)
    i_mem = int(mem_hits[0])
    t_mem = int(steps[i_mem])
    if test_acc[i_mem] >= p.simultaneous_threshold:
        return Detection(outcome="simultaneous", t_mem=t_mem, censored=False, **empty, **common)

    segment = test_acc[i_mem:]
    if len(segment) < 2:
        return Detection(outcome="no_generalization", t_mem=t_mem, censored=True, **empty, **common)
    k, _ = changepoint_l2(segment)
    jump = float(segment[k:].mean() - segment[:k].mean())
    if jump < p.min_jump:
        return Detection(
            outcome="no_generalization", t_mem=t_mem, censored=True, **{**empty, "jump": jump}, **common
        )

    t_grok = int(steps[i_mem + k])
    midpoint = 0.5 * (test_acc[i_mem] + end_test)
    outcome = "grokking" if end_test >= p.gen_threshold else "partial"
    return Detection(
        outcome=outcome,
        t_mem=t_mem,
        t_grok=t_grok,
        censored=False,
        jump=jump,
        t_grok_thr50=threshold_crossing(steps, test_acc, i_mem, midpoint),
        t_grok_logderiv=max_log_derivative(steps, test_acc, i_mem, p.smooth_window),
        **common,
    )


def synthetic_curves(
    steps: np.ndarray,
    t_mem: float,
    t_star: float,
    width: float,
    low: float,
    high: float,
    noise: float,
    rng: np.random.Generator,
    dip: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Синтетические train_acc и test_acc с известным моментом перехода t_star.

    test_acc — сигмоида по log t с серединой в t_star и шириной width плюс шум;
    dip = (начало, длительность, глубина) — кратковременный провал после перехода.
    train_acc — быстрая сигмоида, выходящая на 1.0 к t_mem.
    """
    t = np.maximum(steps.astype(np.float64), 1.0)
    u = np.log(t)
    test = low + (high - low) / (1.0 + np.exp(-(u - math.log(t_star)) / width))
    test = test + rng.normal(0.0, noise, size=len(t))
    if dip is not None:
        start, duration, depth = dip
        in_dip = (t >= start) & (t < start + duration)
        test[in_dip] *= 1.0 - depth
    train = 1.0 / (1.0 + np.exp(-(u - math.log(t_mem / 2.0)) / 0.12))
    return np.clip(train, 0.0, 1.0), np.clip(test, 0.0, 1.0)
