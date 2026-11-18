import pytest
import torch

from grokking.model import BatchedMLP
from grokking.regularization import frac_small, penalty, weight_norms


def constant_model() -> BatchedMLP:
    """Две реплики: веса реплики 0 равны 0.5, реплики 1 — -2; смещения равны 100."""
    model = BatchedMLP([0, 1], d_in=4, d_hidden=3, d_out=2)
    with torch.no_grad():
        for W in model.weights:
            W[0].fill_(0.5)
            W[1].fill_(-2.0)
        for b in model.biases:
            b.fill_(100.0)
    return model


def test_values_on_known_weights_ignore_biases():
    model = constant_model()
    n = 4 * 3 + 3 * 2  # число весов без смещений
    l1, l2sq = weight_norms(model)
    assert torch.allclose(l1, torch.tensor([0.5 * n, 2.0 * n]))
    assert torch.allclose(l2sq, torch.tensor([0.25 * n, 4.0 * n]))
    assert torch.allclose(penalty(model, "l1"), l1)
    assert torch.allclose(penalty(model, "l2"), l2sq)
    assert torch.equal(penalty(model, "none"), torch.zeros(2))


def test_unknown_kind():
    with pytest.raises(ValueError):
        penalty(constant_model(), "l3")


def test_frac_small():
    model = constant_model()
    with torch.no_grad():
        model.weights[0][0].zero_()  # у реплики 0 обнуляем первый слой: 12 из 18 весов
    assert torch.allclose(frac_small(model), torch.tensor([12 / 18, 0.0]))
