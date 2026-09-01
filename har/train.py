"""Model comparison under three evaluation protocols.

For every model we report:
  * random 5-fold CV        -- subjects mixed across folds (the leaky protocol
                               most tutorials use)
  * subject-wise 5-fold CV  -- GroupKFold on subject id; measures
                               generalisation to unseen people
  * official test set       -- the dataset's own subject-disjoint split

The gap between the first two columns is the data-leakage penalty.

Run:  python -m har.train
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from . import data
from .evaluate import RESULTS, SEED, plot_confusion, random_cv, subject_cv, test_metrics

# Features ship pre-normalised to [-1, 1], so no scaler is needed here.
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "SVM (RBF)": SVC(),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
    "KNN (k=9)": KNeighborsClassifier(n_neighbors=9),
}


def main() -> None:
    train, test = data.load()
    rows = []
    best = (None, -1.0, None)

    for name, model in MODELS.items():
        rand = random_cv(model, train.X, train.y)
        subj = subject_cv(model, train.X, train.y, train.subject)
        held = test_metrics(model, train.X, train.y, test.X, test.y)
        rows.append({
            "model": name,
            "random_cv": rand.mean(),
            "random_cv_std": rand.std(),
            "subject_cv": subj.mean(),
            "subject_cv_std": subj.std(),
            "leakage_gap": rand.mean() - subj.mean(),
            "test_accuracy": held["accuracy"],
            "test_macro_f1": held["macro_f1"],
        })
        print(f"{name:22s} random {rand.mean():.4f}  "
              f"subject {subj.mean():.4f}±{subj.std():.3f}  "
              f"test {held['accuracy']:.4f}")
        if held["accuracy"] > best[1]:
            best = (name, held["accuracy"], held)

    df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "model_comparison.csv", index=False)

    best_name, _, best_metrics = best
    plot_confusion(test.y, best_metrics["pred"],
                   RESULTS / "confusion_matrix.png",
                   f"{best_name} on official test split")
    (RESULTS / "best_model.json").write_text(json.dumps({
        "model": best_name,
        "test_accuracy": best_metrics["accuracy"],
        "test_macro_f1": best_metrics["macro_f1"],
        "per_class_f1": best_metrics["per_class_f1"],
    }, indent=2))

    _plot_leakage(df)
    print(f"\nBest on test: {best_name} ({best[1]:.4f}). "
          f"Results written to {RESULTS}/")


def _plot_leakage(df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, df["random_cv"], 0.4, label="Random 5-fold CV (leaky)",
           yerr=df["random_cv_std"], capsize=3)
    ax.bar(x + 0.2, df["subject_cv"], 0.4, label="Subject-wise 5-fold CV",
           yerr=df["subject_cv_std"], capsize=3)
    ax.set_xticks(x, df["model"], rotation=20, ha="right")
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Mixing subjects across folds inflates every model's score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "leakage_gap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
