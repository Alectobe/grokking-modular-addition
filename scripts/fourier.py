"""
Фурье-структура весов первого слоя (доп. анализ, docs/EXPERIMENTS.md §8).

Для скрытого нейрона h веса по входу x1 — вектор W1[x1, h], x1 = 0…q−1. Его дискретное
преобразование Фурье показывает, из каких частот по кругу остатков складывается признак.
Доля мощности в 5 главных частотах (сумма по нейронам) измеряет периодичность признаков:
у случайных весов мощность размазана по всем q//2 частотам, у признаков-синусоид —
сосредоточена в нескольких. Обобщающее решение модульного сложения, по литературе,
строится на таких признаках (Nanda et al., 2023; Gromov, 2023).

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

TOP_K = 5


def spectrum(block: np.ndarray) -> np.ndarray:
    """Доли мощности по частотам k = 1…q//2 (сумма по нейронам; постоянная составляющая отброшена)."""
    power = (np.abs(np.fft.rfft(block, axis=0)[1:]) ** 2).sum(axis=1)
    return power / power.sum()


def top_share(power: np.ndarray, k: int = TOP_K) -> float:
    return float(np.sort(power)[::-1][:k].sum())


def replica_spectra(job, r: int, q: int) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """Спектры по входам x1 и x2 на всех сохранённых эпохах."""
    steps, spec_x1, spec_x2 = [], [], []
    for path in sorted((job.path / "checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        W1 = torch.load(path)["weights.0"][r].numpy()  # (2q, H)
        steps.append(int(path.stem.split("_")[1]))
        spec_x1.append(spectrum(W1[:q]))
        spec_x2.append(spectrum(W1[q : 2 * q]))
    return steps, spec_x1, spec_x2


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

    styles = {("l2", 1e-4): (COLOR_L2, "-", True), ("l2", 3e-6): (COLOR_L2, "--", False),
              ("l2", 0.0): (COLOR_CONTROL, ":", True), ("l1", 3e-6): (COLOR_L1, "-", True),
              ("l1", 1e-4): (COLOR_L1, "--", False)}
    names = {"grokking": "grokking", "partial": "частичное", "no_generalization": "нет обобщения",
             "no_memorization": "нет запоминания", "simultaneous": "без задержки"}
    runs, rows = [], []
    for job in iter_jobs(args.results):
        kind, q = job.config["regularization"]["kind"], job.config["data"]["q"]
        m = job.metrics
        for r in range(len(m["seed"])):
            rho = float(m["rho"][r])
            key = (kind, rho)
            if key not in styles:  # L1 с ρ = 0 совпадает с контролем L2 с ρ = 0
                continue
            d = detect(m["step"], m["train_acc"][:, r], m["test_acc"][:, r])
            steps, s1, s2 = replica_spectra(job, r, q)
            share = [(top_share(a) + top_share(b)) / 2 for a, b in zip(s1, s2)]
            color, ls, show = styles[key]
            label = "Без регуляризации (ρ = 0)" if rho == 0 else f"{kind.upper()}, ρ = {sci_label(rho)}"
            runs.append({"label": f"{label}: {names[d.outcome]}", "color": color, "ls": ls, "steps": steps,
                         "share": share, "t_grok": d.t_grok, "final_power": (s1[-1] + s2[-1]) / 2,
                         "init_power": (s1[0] + s2[0]) / 2, "show_spectrum": show, "order": (kind != "l2", -rho)})
            i_mem, i_grok = nearest(steps, d.t_mem), nearest(steps, d.t_grok)
            top_x1 = sorted(int(k) + 1 for k in np.argsort(s1[-1])[::-1][:TOP_K])
            top_x2 = sorted(int(k) + 1 for k in np.argsort(s2[-1])[::-1][:TOP_K])
            rows.append({
                "запуск": label, "исход": names[d.outcome], "t_mem": d.t_mem, "t_grok": d.t_grok,
                "итоговая test_acc": f"{100 * d.end_test_acc:.1f}%",
                "доля 5 частот: начало": f"{100 * share[0]:.0f}%",
                "около t_mem": f"{100 * share[i_mem]:.0f}% (эп. {steps[i_mem]})" if i_mem is not None else "—",
                "около t_grok": f"{100 * share[i_grok]:.0f}% (эп. {steps[i_grok]})" if i_grok is not None else "—",
                "конец": f"{100 * share[-1]:.0f}%",
                "главные частоты x1": ", ".join(map(str, top_x1)), "главные частоты x2": ", ".join(map(str, top_x2)),
            })
    runs.sort(key=lambda run: run["order"])
    plot_fourier(runs, args.figures / "fourier.png")
    table = pd.DataFrame(rows)
    save_table(table, args.tables / "fourier")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
