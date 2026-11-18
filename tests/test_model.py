import json
from argparse import Namespace

import numpy as np
import torch

from grokking.config import Config, DataConfig, ModelConfig, OptimizerConfig, RegularizationConfig
from grokking.model import BatchedMLP
from grokking.regularization import penalty
from grokking.train import cross_entropy_per_replica, train_job


def small_config(rhos, seeds, kind="l2", epochs=30) -> Config:
    return Config(
        experiment_name="test",
        output_root="unused",
        data=DataConfig(q=11, train_frac=0.4, split_seed=0),
        model=ModelConfig(d_hidden=16, n_hidden_layers=1, activation="relu"),
        optimizer=OptimizerConfig(name="adam", lr=1e-2, betas=(0.9, 0.98), eps=1e-8),
        regularization=RegularizationConfig(kind=kind, rho=tuple(rhos), apply_to="weights"),
        epochs=epochs,
        dtype="float32",
        eval_every=5,
        checkpoint_steps=(),
        seeds=tuple(seeds),
        device="cpu",
        deterministic=True,
        threads=2,
    )


def test_output_shape():
    model = BatchedMLP([0, 1, 2], d_in=14, d_hidden=8, d_out=7)
    assert model(torch.zeros(5, 14)).shape == (3, 5, 7)


def test_replica_weights_depend_only_on_seed():
    a = BatchedMLP([3, 5], d_in=14, d_hidden=8, d_out=7)
    b = BatchedMLP([5], d_in=14, d_hidden=8, d_out=7)
    for Wa, Wb in zip(a.weight_matrices(), b.weight_matrices()):
        assert torch.equal(Wa[1], Wb[0])


def test_replicas_train_independently():
    """R = 3 в одном батче против трёх отдельных запусков с R = 1."""
    torch.manual_seed(0)
    X = torch.nn.functional.one_hot(torch.randint(0, 7, (40,)), 14).float()
    Y = torch.nn.functional.one_hot(torch.randint(0, 7, (40,)), 7).float()
    seeds, rhos = [0, 1, 2], [0.0, 1e-3, 1e-2]

    def train(seed_list, rho_list):
        model = BatchedMLP(seed_list, 14, 8, 7)
        rho = torch.tensor(rho_list)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(50):
            opt.zero_grad()
            loss = (cross_entropy_per_replica(model(X), Y) + rho * penalty(model, "l2")).sum()
            loss.backward()
            opt.step()
        return model

    batched = train(seeds, rhos)
    for r in range(3):
        single = train([seeds[r]], [rhos[r]])
        for Wb, Ws in zip(batched.weight_matrices(), single.weight_matrices()):
            assert torch.allclose(Wb[r], Ws[0], atol=1e-5)


def test_train_job_matches_pilot(tmp_path):
    """Ядро повторяет пилот бит в бит: числа из docs/PILOT_RESULTS.md остаются валидными."""
    import pilot_grokking

    cfg = small_config(rhos=[1e-4, 1e-4, 0.0, 0.0], seeds=[0, 1, 0, 1])
    ours = train_job(cfg, tmp_path / "ours")

    args = Namespace(
        q=11, train_frac=0.4, split_seed=0, hidden=16, layers=1, optimizer="adam", reg="l2", lr=1e-2,
        rhos=[1e-4, 0.0], seeds=[0, 1], steps=30, eval_every=5, threads=2, out=str(tmp_path / "pilot"),
    )
    pilot_grokking.run(args)
    with np.load(tmp_path / "pilot" / "metrics.npz") as pilot:
        for key in ["step", "train_loss", "test_loss", "train_acc", "test_acc", "l1_norm", "l2_norm", "frac_small"]:
            assert np.array_equal(ours[key], pilot[key]), key

    meta = json.loads((tmp_path / "ours" / "meta.json").read_text())
    assert meta["status"] == "done"
