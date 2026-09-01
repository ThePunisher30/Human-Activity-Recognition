"""Feature redundancy analysis and a selection curve.

The 561 engineered features are heavily redundant: some columns are exact
duplicates and thousands of pairs are near-perfectly correlated. This module
quantifies that and shows how accuracy behaves as the feature set shrinks to a
fraction of its size -- which is what matters for running on a phone.

Run:  python -m har.features
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import data
from .evaluate import RESULTS, SEED

FEATURE_COUNTS = [10, 25, 50, 100, 200, 561]


def redundancy_report(X: np.ndarray, names: list[str]) -> dict:
    n_unique_cols = np.unique(X.round(6), axis=1).shape[1]
    counts = pd.Series(names).value_counts()
    corr = np.corrcoef(X[::5].T)
    upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
    return {
        "n_features": X.shape[1],
        "n_unique_columns": int(n_unique_cols),
        "n_duplicate_names": int((counts > 1).sum()),
        "pairs_above_r95": int((upper > 0.95).sum()),
        "pairs_total": int(upper.size),
    }


def main() -> None:
    train, test = data.load()
    names = data.feature_names()

    report = redundancy_report(train.X, names)
    print("Redundancy:", json.dumps(report, indent=2))

    # Rank features once with a random forest, then retrain a logistic
    # regression on progressively larger prefixes of that ranking.
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(train.X, train.y)
    order = np.argsort(rf.feature_importances_)[::-1]

    rows = []
    for k in FEATURE_COUNTS:
        cols = order[:k]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(train.X[:, cols], train.y)
        acc = float((clf.predict(test.X[:, cols]) == test.y).mean())
        rows.append({"n_features": k, "test_accuracy": acc})
        print(f"top {k:3d} features -> test accuracy {acc:.4f}")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "feature_selection_curve.csv", index=False)
    (RESULTS / "feature_redundancy.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "feature": [names[i] for i in order[:25]],
        "importance": rf.feature_importances_[order[:25]],
    }).to_csv(RESULTS / "top_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["n_features"], df["test_accuracy"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Number of features (RF-importance ranked, log scale)")
    ax.set_ylabel("Test accuracy (logistic regression)")
    ax.set_title("Most of the 561 features are redundant")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_selection_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
