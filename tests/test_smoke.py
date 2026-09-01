"""Fast sanity checks. Run: python -m pytest tests/ (dataset must be in data/)."""

import numpy as np
import pytest

from har import ACTIVITY_NAMES, data


@pytest.fixture(scope="module")
def splits():
    try:
        return data.load()
    except Exception:
        pytest.skip("dataset not available in data/")


def test_shapes(splits):
    train, test = splits
    assert train.X.shape == (7352, 561)
    assert test.X.shape == (2947, 561)
    assert len(data.feature_names()) == 561


def test_labels_and_subjects(splits):
    train, test = splits
    assert set(np.unique(train.y)) == set(ACTIVITY_NAMES)
    # The official split is subject-disjoint; everything downstream relies on it.
    assert not set(train.subject) & set(test.subject)


def test_features_normalised(splits):
    train, _ = splits
    assert train.X.min() >= -1.001 and train.X.max() <= 1.001
    assert not np.isnan(train.X).any()


def test_two_stage_roundtrip(splits):
    from har.two_stage import TwoStageClassifier
    train, _ = splits
    sub = slice(0, 600)
    clf = TwoStageClassifier().fit(train.X[sub], train.y[sub])
    pred = clf.predict(train.X[sub])
    assert set(pred) <= set(ACTIVITY_NAMES)
    assert (pred == train.y[sub]).mean() > 0.8
