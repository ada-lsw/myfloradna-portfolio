import pandas as pd
import pytest

from myflora.model.split import split_batches


def _dataset(n: int) -> pd.DataFrame:
    return pd.DataFrame({"batch_id": [f"b{i}" for i in range(n)], "x": range(n)})


def test_split_proportions_are_approximately_correct():
    train, test = split_batches(_dataset(100), test_fraction=0.2, master_seed=1)
    assert len(test) == 20
    assert len(train) == 80


def test_split_covers_every_row_with_no_overlap():
    dataset = _dataset(50)
    train, test = split_batches(dataset, test_fraction=0.3, master_seed=2)

    train_ids = set(train["batch_id"])
    test_ids = set(test["batch_id"])

    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(dataset["batch_id"])


def test_split_is_reproducible_given_same_seed():
    dataset = _dataset(40)
    train_a, test_a = split_batches(dataset, test_fraction=0.25, master_seed=5)
    train_b, test_b = split_batches(dataset, test_fraction=0.25, master_seed=5)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_different_seeds_give_different_splits():
    dataset = _dataset(40)
    _, test_a = split_batches(dataset, test_fraction=0.25, master_seed=1)
    _, test_b = split_batches(dataset, test_fraction=0.25, master_seed=2)

    assert set(test_a["batch_id"]) != set(test_b["batch_id"])


def test_at_least_one_test_row_for_small_datasets():
    train, test = split_batches(_dataset(3), test_fraction=0.1, master_seed=1)
    assert len(test) >= 1
    assert len(train) >= 1


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_rejects_invalid_test_fraction(bad_fraction):
    with pytest.raises(ValueError):
        split_batches(_dataset(10), test_fraction=bad_fraction, master_seed=1)


def test_rejects_dataset_too_small_to_split():
    with pytest.raises(ValueError):
        split_batches(_dataset(1), test_fraction=0.2, master_seed=1)
