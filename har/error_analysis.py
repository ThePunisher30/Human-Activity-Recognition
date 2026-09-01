"""Where the errors live: per-subject accuracy and the SITTING/STANDING axis.

Accuracy averaged over windows hides two things this script surfaces:
  * some *people* are much harder than others (per-subject accuracy on the
    official test split spans a wide range), and
  * most residual error is one confusion pair, SITTING vs STANDING.

Run:  python -m har.error_analysis
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import ACTIVITY_NAMES, data
from .evaluate import RESULTS


def main() -> None:
    train, test = data.load()
    clf = LogisticRegression(max_iter=2000).fit(train.X, train.y)
    pred = clf.predict(test.X)
    correct = pred == test.y

    per_subject = (
        pd.DataFrame({"subject": test.subject, "correct": correct})
        .groupby("subject")["correct"].agg(["mean", "size"])
        .rename(columns={"mean": "accuracy", "size": "n_windows"})
        .sort_values("accuracy")
    )
    RESULTS.mkdir(exist_ok=True)
    per_subject.to_csv(RESULTS / "per_subject_accuracy.csv")
    print(per_subject)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(per_subject.index.astype(str), per_subject["accuracy"])
    ax.axhline(correct.mean(), ls="--", c="gray",
               label=f"overall {correct.mean():.3f}")
    ax.set_xlabel("Test subject")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("The same model, very different people")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "per_subject_accuracy.png", dpi=150)
    plt.close(fig)

    # How much of the total error is each confusion pair responsible for?
    errors = pd.DataFrame({
        "true": [ACTIVITY_NAMES[k] for k in test.y[~correct]],
        "pred": [ACTIVITY_NAMES[k] for k in pred[~correct]],
    })
    pair_counts = (
        errors.value_counts().rename("count").reset_index()
        .assign(share=lambda d: d["count"] / (~correct).sum())
    )
    pair_counts.to_csv(RESULTS / "error_pairs.csv", index=False)
    print("\nError share by confusion pair:")
    print(pair_counts.head(6).to_string(index=False))

    summary = {
        "overall_accuracy": float(correct.mean()),
        "worst_subject": int(per_subject.index[0]),
        "worst_subject_accuracy": float(per_subject["accuracy"].iloc[0]),
        "best_subject_accuracy": float(per_subject["accuracy"].iloc[-1]),
        "top_error_pair": (
            f"{pair_counts['true'].iloc[0]} -> {pair_counts['pred'].iloc[0]}"
        ),
        "top_error_pair_share": float(pair_counts["share"].iloc[0]),
    }
    (RESULTS / "error_analysis.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
