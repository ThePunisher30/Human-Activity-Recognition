"""Hyperparameter tuning with a subject-wise search protocol.

Tuning is model selection, so it can leak exactly like evaluation can: if the
search's inner CV mixes subjects across folds, it picks hyperparameters that
exploit person-specific patterns. Here every search scores candidates with
GroupKFold on subject id, the winner is refit on the full training split, and
the official subject-disjoint test set is touched exactly once per model.

Finally, the top tuned models are combined into a soft-voting ensemble.

Run:  python -m har.tune            (full search, ~30-60 min on CPU)
      python -m har.tune --quick    (smaller budget, for smoke-testing)
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from . import data
from .evaluate import RESULTS, SEED, plot_confusion

SEARCHES = {
    "LogisticRegression": (
        LogisticRegression(max_iter=5000),
        {"C": loguniform(1e-3, 1e2)},
        15,
    ),
    "LinearSVC": (
        LinearSVC(max_iter=5000),
        {"C": loguniform(1e-4, 1e1)},
        15,
    ),
    "SVM (RBF)": (
        SVC(),
        {"C": loguniform(1e-1, 1e3), "gamma": loguniform(1e-4, 1e-1)},
        12,
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED),
        {
            "n_estimators": [300, 500],
            "max_features": ["sqrt", 0.1, 0.3],
            "min_samples_leaf": [1, 2, 4],
        },
        10,
    ),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {
            "learning_rate": loguniform(3e-2, 3e-1),
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": loguniform(1e-3, 1e1),
            "max_iter": [200, 400],
        },
        10,
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [5, 9, 15, 21, 31],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        12,
    ),
}

ENSEMBLE_MEMBERS = ["LogisticRegression", "SVM (RBF)", "HistGradientBoosting"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="tiny search budget, for smoke tests")
    args = parser.parse_args()

    train, test = data.load()
    cv = GroupKFold(5)
    rows, fitted, preds = [], {}, {}

    for name, (model, space, n_iter) in SEARCHES.items():
        search = RandomizedSearchCV(
            model, space,
            n_iter=2 if args.quick else n_iter,
            cv=cv, n_jobs=-1, random_state=SEED, refit=True,
        )
        search.fit(train.X, train.y, groups=train.subject)
        pred = search.best_estimator_.predict(test.X)
        acc = float((pred == test.y).mean())
        rows.append({
            "model": name,
            "subject_cv": search.best_score_,
            "test_accuracy": acc,
            "test_macro_f1": float(f1_score(test.y, pred, average="macro")),
            "best_params": json.dumps(
                {k: (round(v, 5) if isinstance(v, float) else v)
                 for k, v in search.best_params_.items()}
            ),
        })
        fitted[name], preds[name] = search.best_estimator_, pred
        print(f"{name:22s} subject-CV {search.best_score_:.4f}  "
              f"test {acc:.4f}  {rows[-1]['best_params']}")

    # Soft-voting ensemble of the strongest tuned models. SVC needs
    # probability=True for soft voting; rebuild it with the tuned params.
    members = []
    for name in ENSEMBLE_MEMBERS:
        est = fitted[name]
        if isinstance(est, SVC):
            # Soft voting needs predict_proba; calibrate the tuned SVC.
            est = CalibratedClassifierCV(SVC(**est.get_params()), ensemble=False)
        members.append((name, est))
    ensemble = VotingClassifier(members, voting="soft", n_jobs=-1)
    ensemble.fit(train.X, train.y)
    pred = ensemble.predict(test.X)
    acc = float((pred == test.y).mean())
    rows.append({
        "model": "Ensemble (soft vote)",
        "subject_cv": np.nan,
        "test_accuracy": acc,
        "test_macro_f1": float(f1_score(test.y, pred, average="macro")),
        "best_params": json.dumps({"members": ENSEMBLE_MEMBERS}),
    })
    preds["Ensemble (soft vote)"] = pred
    print(f"{'Ensemble (soft vote)':22s} test {acc:.4f}")

    df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "tuning.csv", index=False)

    best = df.iloc[0]
    plot_confusion(test.y, preds[best["model"]],
                   RESULTS / "confusion_tuned_best.png",
                   f"{best['model']} (tuned) on official test split")
    (RESULTS / "tuned_best.json").write_text(json.dumps({
        "model": best["model"],
        "test_accuracy": best["test_accuracy"],
        "test_macro_f1": best["test_macro_f1"],
        "params": json.loads(best["best_params"]),
    }, indent=2))
    print(f"\nBest after tuning: {best['model']} "
          f"({best['test_accuracy']:.4f}). Results in {RESULTS}/tuning.csv")


if __name__ == "__main__":
    main()
