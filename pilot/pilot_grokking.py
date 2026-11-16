"""
Пилотный прогон: проверка, что маленькая полносвязная сеть на задаче
y = (x1 + x2) mod q демонстрирует отложенное обобщение (grokking).

Одним запуском обучается R независимых реплик сети (разные seed и/или rho).
Веса реплик лежат в общих тензорах с ведущей осью R. Реплики не влияют
друг на друга, потому что:
  * итоговый лосс — сумма лоссов реплик, значит градиент по весам реплики r
    зависит только от лосса реплики r;
  * Adam обновляет каждый элемент тензора независимо;
  * глобальный клиппинг градиента не используется (он связал бы реплики).

Обучение полнобатчевое: один шаг оптимизатора = одна эпоха.

Пример запуска:
    python pilot/pilot_grokking.py --reg l2 --lr 1e-3 --rhos 1e-4 1e-3 --seeds 0 1 2 \
        --steps 20000 --out pilot_runs/l2_lr1e-3
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def make_dataset(q: int, train_frac: float, split_seed: int):
    """Строит все q*q пар и делит их на train/test фиксированным разбиением.

    Вход: конкатенация one-hot(x1) и one-hot(x2), длина 2q.
    Цель: one-hot(y) длины q, где y = (x1 + x2) mod q.
    """
    x1, x2 = torch.meshgrid(torch.arange(q), torch.arange(q), indexing="ij")
    x1, x2 = x1.reshape(-1), x2.reshape(-1)
    y = (x1 + x2) % q
    X = torch.cat([F.one_hot(x1, q), F.one_hot(x2, q)], dim=1).float()
    Y = F.one_hot(y, q).float()

    # Разбиение зависит только от split_seed и одинаково для всех реплик
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(q * q, generator=g)
    n_train = int(round(train_frac * q * q))
    idx_train, idx_test = perm[:n_train], perm[n_train:]
    return X[idx_train], Y[idx_train], X[idx_test], Y[idx_test]


class BatchedMLP(torch.nn.Module):
    """R независимых MLP вида d_in -> H -> ... -> H -> d_out с ReLU.

    Веса слоя l хранятся тензором (R, d_in_l, d_out_l), смещения — (R, d_out_l).
    Инициализация каждой реплики — как у nn.Linear (равномерная в
    [-1/sqrt(fan_in), 1/sqrt(fan_in)]), но со своим генератором, поэтому
    веса реплики зависят только от её seed.
    """

    def __init__(self, seeds, d_in, d_hidden, d_out, n_hidden_layers):
        super().__init__()
        n_replicas = len(seeds)
        dims = [d_in] + [d_hidden] * n_hidden_layers + [d_out]
        weights = [torch.empty(n_replicas, a, b) for a, b in zip(dims[:-1], dims[1:])]
        biases = [torch.empty(n_replicas, b) for b in dims[1:]]

        for r, seed in enumerate(seeds):
            g = torch.Generator().manual_seed(int(seed))
            for W, b, fan_in in zip(weights, biases, dims[:-1]):
                bound = 1.0 / math.sqrt(fan_in)
                W[r].uniform_(-bound, bound, generator=g)
                b[r].uniform_(-bound, bound, generator=g)

        self.weights = torch.nn.ParameterList([torch.nn.Parameter(W) for W in weights])
        self.biases = torch.nn.ParameterList([torch.nn.Parameter(b) for b in biases])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, d_in), общий для всех реплик. Возвращает логиты (R, N, d_out)."""
        n_replicas = self.weights[0].shape[0]
        h = X.unsqueeze(0).expand(n_replicas, -1, -1)
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b.unsqueeze(1), h, W)
            if i < n_layers - 1:
                h = torch.relu(h)
        return h


def cross_entropy_per_replica(logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Перекрёстная энтропия с one-hot целью, усреднённая по объектам. Форма (R,)."""
    return -(Y.unsqueeze(0) * torch.log_softmax(logits, dim=-1)).sum(-1).mean(-1)


def accuracy_per_replica(logits: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Доля верных ответов (argmax логитов совпал с классом). Форма (R,)."""
    return (logits.argmax(-1) == Y.argmax(-1).unsqueeze(0)).float().mean(-1)


def weight_norms(model: BatchedMLP):
    """L1- и квадрат L2-нормы всех матриц весов (без смещений) для каждой реплики."""
    l1 = sum(W.abs().sum(dim=(1, 2)) for W in model.weights)
    l2sq = sum((W ** 2).sum(dim=(1, 2)) for W in model.weights)
    return l1, l2sq


def first_step_at_least(steps: np.ndarray, values: np.ndarray, threshold: float):
    """Первый шаг, на котором значение >= threshold. None, если такого нет."""
    hit = np.nonzero(values >= threshold)[0]
    return int(steps[hit[0]]) if len(hit) else None


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, Y_tr, X_te, Y_te = make_dataset(args.q, args.train_frac, args.split_seed)
    X_tr, Y_tr, X_te, Y_te = (t.to(device) for t in (X_tr, Y_tr, X_te, Y_te))

    # Декартово произведение rho x seed: каждая пара — отдельная реплика
    replica_rho = [rho for rho in args.rhos for _ in args.seeds]
    replica_seed = [seed for _ in args.rhos for seed in args.seeds]
    rho = torch.tensor(replica_rho, dtype=torch.float32, device=device)

    model = BatchedMLP(replica_seed, 2 * args.q, args.hidden, args.q, args.layers).to(device)

    # Регуляризация задаётся только явным штрафом в лоссе, у оптимизаторов weight decay = 0
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Ключи совпадают с контрактом metrics.npz из docs/ARCHITECTURE.md §5
    keys = ["step", "train_loss", "test_loss", "train_acc", "test_acc",
            "penalty", "l1_norm", "l2_norm", "frac_small"]
    log = {k: [] for k in keys}

    def penalty_value(l1, l2sq):
        """R(theta) без множителя rho для выбранного типа регуляризации."""
        if args.reg == "l1":
            return l1
        if args.reg == "l2":
            return l2sq
        return torch.zeros_like(l1)

    def evaluate(step):
        with torch.no_grad():
            logits_tr, logits_te = model(X_tr), model(X_te)
            l1, l2sq = weight_norms(model)
            n_small = sum((W.abs() < 1e-3).sum(dim=(1, 2)) for W in model.weights)
            n_total = sum(W[0].numel() for W in model.weights)
            log["step"].append(step)
            log["train_loss"].append(cross_entropy_per_replica(logits_tr, Y_tr).cpu().numpy())
            log["test_loss"].append(cross_entropy_per_replica(logits_te, Y_te).cpu().numpy())
            log["train_acc"].append(accuracy_per_replica(logits_tr, Y_tr).cpu().numpy())
            log["test_acc"].append(accuracy_per_replica(logits_te, Y_te).cpu().numpy())
            log["penalty"].append(penalty_value(l1, l2sq).cpu().numpy())
            log["l1_norm"].append(l1.cpu().numpy())
            log["l2_norm"].append(l2sq.sqrt().cpu().numpy())
            log["frac_small"].append((n_small.float() / n_total).cpu().numpy())

    t0 = time.time()
    evaluate(0)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ce = cross_entropy_per_replica(model(X_tr), Y_tr)
        l1, l2sq = weight_norms(model)
        penalty = penalty_value(l1, l2sq)
        loss = (ce + rho * penalty).sum()
        loss.backward()
        optimizer.step()
        if step % args.eval_every == 0:
            evaluate(step)
    elapsed = time.time() - t0

    arrays = {k: np.asarray(v) for k, v in log.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "metrics.npz", rho=np.asarray(replica_rho), seed=np.asarray(replica_seed), **arrays)

    # Простейшие пороговые отметки для пилота (финальный критерий — в docs/EXPERIMENTS.md)
    summary = []
    for r in range(len(replica_seed)):
        t_mem = first_step_at_least(arrays["step"], arrays["train_acc"][:, r], 0.99)
        t_gen = first_step_at_least(arrays["step"], arrays["test_acc"][:, r], 0.95)
        summary.append(
            {
                "rho": replica_rho[r],
                "seed": replica_seed[r],
                "t_mem_99": t_mem,
                "t_gen_95": t_gen,
                "final_train_acc": float(arrays["train_acc"][-1, r]),
                "final_test_acc": float(arrays["test_acc"][-1, r]),
                "test_acc_at_t_mem": None
                if t_mem is None
                else float(arrays["test_acc"][arrays["step"] == t_mem, r][0]),
            }
        )
    meta = {"args": vars(args), "elapsed_sec": elapsed, "device": str(device), "summary": summary}
    (out / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Кривые точности и лосса, ось эпох — логарифмическая
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    steps = np.maximum(arrays["step"], 1)
    for r in range(len(replica_seed)):
        label = f"rho={replica_rho[r]:g}, seed={replica_seed[r]}"
        line = axes[0].plot(steps, arrays["train_acc"][:, r], lw=1, ls="--")[0]
        axes[0].plot(steps, arrays["test_acc"][:, r], lw=1.5, color=line.get_color(), label=label)
        axes[1].plot(steps, arrays["train_loss"][:, r], lw=1, ls="--", color=line.get_color())
        axes[1].plot(steps, arrays["test_loss"][:, r], lw=1.5, color=line.get_color())
    axes[0].set(xscale="log", xlabel="эпоха", ylabel="точность", title="точность (пунктир — train, сплошная — test)")
    axes[1].set(xscale="log", yscale="log", xlabel="эпоха", ylabel="CE", title="функция потерь (пунктир — train)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"q={args.q}, frac={args.train_frac}, {args.optimizer}, reg={args.reg}, lr={args.lr:g}")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)

    print(f"{out}: {elapsed:.1f} c, устройство {device}")
    for s in summary:
        print(
            f"  rho={s['rho']:<8g} seed={s['seed']:<3d} t_mem={s['t_mem_99']} t_gen={s['t_gen_95']} "
            f"test@t_mem={s['test_acc_at_t_mem']} final train/test="
            f"{s['final_train_acc']:.3f}/{s['final_test_acc']:.3f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Пилотный прогон grokking на сложении по модулю q")
    p.add_argument("--q", type=int, default=97)
    p.add_argument("--train-frac", type=float, default=0.4)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1, help="число скрытых слоёв")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--reg", choices=["none", "l1", "l2"], default="l2")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rhos", type=float, nargs="+", default=[1e-4])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=str, default="pilot_runs/run")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
