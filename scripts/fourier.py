"""
Фурье-структура весов первого слоя (доп. анализ, docs/EXPERIMENTS.md §8).

Веса скрытого нейрона h по входу x1 — вектор W1[x1, h], x1 = 0…q−1. Его дискретное
преобразование Фурье показывает, из каких частот по кругу остатков складывается признак.

Периодичность нейронов — доля мощности нейрона в его главной частоте, в среднем по нейронам
с весами, пропорциональными мощности нейрона. У случайных весов это ~0.1 (мощность размазана
по q//2 = 48 частотам), у нейрона-синусоиды — 1. Обобщающее решение модульного сложения,
по литературе, строится на таких признаках (Nanda et al., 2023; Gromov, 2023).

Живой нейрон — у которого произведение норм входных и выходных весов больше 1% от максимума.

Пример:
    python scripts/run_grid.py --grid configs/fourier.yaml
    python scripts/fourier.py --results results/fourier --figures report/figures --tables report/tables
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking.detection import detect
from grokking.plotting import COLOR_CONTROL, COLOR_L1, COLOR_L2, plot_fourier, sci_label
from grokking.report import save_table
from grokking.storage import iter_jobs

ALIVE_FRACTION = 0.01


def periodicity(block: np.ndarray) -> tuple[float, np.ndarray]:
    """Взвешенная по мощности доля главной частоты нейронов и главные частоты (k = 1…q//2)."""
    power = np.abs(np.fft.rfft(block, axis=0)[1:]) ** 2  # (q//2, H), без постоянной составляющей
    total = power.sum(axis=0)
    weights = total / total.sum()
    share = power.max(axis=0) / np.maximum(total, 1e-30)
    return float((share * weights).sum()), power.argmax(axis=0) + 1


def alive_neurons(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    strength = np.linalg.norm(W1, axis=0) * np.linalg.norm(W2, axis=1)
    return strength > ALIVE_FRACTION * strength.max()


def checkpoints(job) -> list[Path]:
    return sorted((job.path / "checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def nearest(steps: list[int], t: float | None) -> int | None:
    """Индекс сохранённой эпохи, ближайшей к t по логарифмической шкале."""
    if t is None:
        return None
    return int(np.argmin(np.abs(np.log(np.maximum(steps, 1)) - np.log(t))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Фурье-структура весов первого слоя")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()

    styles = {("l2", 1e-4): (COLOR_L2, "-"), ("l2", 3e-6): (COLOR_L2, "--"), ("l2", 0.0): (COLOR_CONTROL, ":"),
              ("l1", 3e-6): (COLOR_L1, "-"), ("l1", 1e-4): (COLOR_L1, "--")}
    names = {"grokking": "grokking", "partial": "частичное", "no_generalization": "нет обобщения",
             "no_memorization": "нет запоминания", "simultaneous": "без задержки"}
    runs, rows, example = [], [], None
    for job in iter_jobs(args.results):
        kind, q = job.config["regularization"]["kind"], job.config["data"]["q"]
        m = job.metrics
        paths = checkpoints(job)
        states = [torch.load(path) for path in paths]
        steps = [int(path.stem.split("_")[1]) for path in paths]
        for r in range(len(m["seed"])):
            rho = float(m["rho"][r])
            if (kind, rho) not in styles:  # L1 с ρ = 0 — то же обучение, что и контроль L2 с ρ = 0
                continue
            d = detect(m["step"], m["train_acc"][:, r], m["test_acc"][:, r])
            per, alive, n_freq = [], [], []
            for state in states:
                W1, W2 = state["weights.0"][r].numpy(), state["weights.1"][r].numpy()
                (p1, f1), (p2, _) = periodicity(W1[:q]), periodicity(W1[q : 2 * q])
                live = alive_neurons(W1, W2)
                per.append((p1 + p2) / 2)
                alive.append(int(live.sum()))
                n_freq.append(len(set(f1[live])))
            label = "Без регуляризации (ρ = 0)" if rho == 0 else f"{kind.upper()}, ρ = {sci_label(rho)}"
            color, ls = styles[(kind, rho)]
            if d.outcome != "no_memorization":  # у сети без запоминания веса схлопываются — на графике шум
                runs.append({"label": f"{label}: {names[d.outcome]}", "color": color, "ls": ls, "steps": steps,
                             "periodicity": per, "alive": alive, "t_grok": d.t_grok, "order": (kind != "l2", -rho)})
            if (kind, rho) == ("l2", 1e-4):
                W_end, W_init = states[-1]["weights.0"][r].numpy()[:q], states[0]["weights.0"][r].numpy()[:q]
                strongest = np.argsort(np.linalg.norm(W_end, axis=0))[::-1][:3]

                def neuron_spectrum(W, h):
                    power = np.abs(np.fft.rfft(W[:, h]))[1:] ** 2
                    return power / power.sum()

                spectra = [neuron_spectrum(W_end, h) for h in strongest]
                example = {"label": f"Спектры трёх самых сильных нейронов ({label}, конец)", "spectra": spectra,
                           "names": [f"нейрон {h}: главная частота k = {int(np.argmax(sp)) + 1}"
                                     for h, sp in zip(strongest, spectra)],
                           "init_spectrum": neuron_spectrum(W_init, strongest[0])}
            i_mem, i_grok = nearest(steps, d.t_mem), nearest(steps, d.t_grok)
            rows.append({
                "запуск": label, "исход": names[d.outcome], "t_mem": d.t_mem, "t_grok": d.t_grok,
                "итоговая test_acc": f"{100 * d.end_test_acc:.1f}%",
                "периодичность: начало": f"{per[0]:.2f}",
                "около t_mem": f"{per[i_mem]:.2f} (эп. {steps[i_mem]})" if i_mem is not None else "—",
                "около t_grok": f"{per[i_grok]:.2f} (эп. {steps[i_grok]})" if i_grok is not None else "—",
                "конец": f"{per[-1]:.2f}",
                "живых нейронов в конце": alive[-1], "разных главных частот": n_freq[-1],
            })
    runs.sort(key=lambda run: run["order"])
    plot_fourier(runs, example, args.figures / "fourier.png")
    table = pd.DataFrame(rows).sort_values("запуск").reset_index(drop=True)
    save_table(table, args.tables / "fourier")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
