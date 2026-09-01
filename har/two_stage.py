"""Two-stage (hierarchical) classifier.

Nearly all of the flat models' errors are SITTING vs STANDING -- static
postures that differ only in gravity orientation, not motion. So we split the
problem:

  stage 1: static vs dynamic (nearly perfect -- motion energy separates them)
  stage 2a: which dynamic activity (walking / upstairs / downstairs)
  stage 2b: which static posture (sitting / standing / laying)

Stage 2b gets its own classifier that can focus entirely on the orientation
features instead of sharing capacity with the motion classes.

Run:  python -m har.two_stage
"""

import json

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from . import ACTIVITY_NAMES, DYNAMIC, data
from .evaluate import RESULTS, plot_confusion


class TwoStageClassifier:
    def __init__(self):
        self.gate = LogisticRegression(max_iter=2000)
        self.dynamic = SVC()
        self.static = SVC()

    def fit(self, X, y):
        is_dyn = np.isin(y, DYNAMIC)
        self.gate.fit(X, is_dyn)
        self.dynamic.fit(X[is_dyn], y[is_dyn])
        self.static.fit(X[~is_dyn], y[~is_dyn])
        return self

    def predict(self, X):
        pred = np.empty(len(X), dtype=int)
        is_dyn = self.gate.predict(X).astype(bool)
        if is_dyn.any():
            pred[is_dyn] = self.dynamic.predict(X[is_dyn])
        if (~is_dyn).any():
            pred[~is_dyn] = self.static.predict(X[~is_dyn])
        return pred


def main() -> None:
    train, test = data.load()

    flat = clone(SVC()).fit(train.X, train.y)
    flat_pred = flat.predict(test.X)

    two = TwoStageClassifier().fit(train.X, train.y)
    two_pred = two.predict(test.X)

    gate_acc = float(
        (np.isin(two_pred, DYNAMIC) == np.isin(test.y, DYNAMIC)).mean()
    )

    result = {"stage1_static_vs_dynamic_accuracy": gate_acc}
    for name, pred in [("flat_svm", flat_pred), ("two_stage", two_pred)]:
        result[name] = {
            "accuracy": float((pred == test.y).mean()),
            "macro_f1": float(f1_score(test.y, pred, average="macro")),
            "per_class_f1": {
                ACTIVITY_NAMES[k]: float(v)
                for k, v in zip(sorted(ACTIVITY_NAMES),
                                f1_score(test.y, pred, average=None))
            },
        }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "two_stage.json").write_text(json.dumps(result, indent=2))
    plot_confusion(test.y, two_pred, RESULTS / "confusion_two_stage.png",
                   "Two-stage classifier on official test split")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
