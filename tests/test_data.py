import pytest
import torch

from grokking.data import make_modular_addition


def test_sizes_and_split_cover_all_pairs():
    q = 7
    data = make_modular_addition(q, 0.4, split_seed=0)
    assert data.X_train.shape == (20, 2 * q)  # round(0.4 * 49) = 20
    assert data.X_test.shape == (29, 2 * q)
    all_pairs = {tuple(p) for p in torch.cat([data.pairs_train, data.pairs_test]).tolist()}
    train_pairs = {tuple(p) for p in data.pairs_train.tolist()}
    test_pairs = {tuple(p) for p in data.pairs_test.tolist()}
    assert len(all_pairs) == q * q
    assert not train_pairs & test_pairs


def test_one_hot_and_labels():
    q = 11
    data = make_modular_addition(q, 0.5, split_seed=1)
    for X, Y, pairs in [(data.X_train, data.Y_train, data.pairs_train), (data.X_test, data.Y_test, data.pairs_test)]:
        assert torch.all(X[:, :q].sum(1) == 1) and torch.all(X[:, q:].sum(1) == 1)
        assert torch.all(Y.sum(1) == 1)
        assert torch.equal(X[:, :q].argmax(1), pairs[:, 0])
        assert torch.equal(X[:, q:].argmax(1), pairs[:, 1])
        assert torch.equal(Y.argmax(1), (pairs[:, 0] + pairs[:, 1]) % q)


def test_split_depends_only_on_split_seed():
    a = make_modular_addition(7, 0.4, split_seed=3)
    b = make_modular_addition(7, 0.4, split_seed=3)
    c = make_modular_addition(7, 0.4, split_seed=4)
    assert torch.equal(a.pairs_train, b.pairs_train)
    assert not torch.equal(a.pairs_train, c.pairs_train)


@pytest.mark.parametrize("q, frac", [(8, 0.4), (7, 0.0), (7, 1.0)])
def test_invalid_arguments(q, frac):
    with pytest.raises(ValueError):
        make_modular_addition(q, frac, split_seed=0)
