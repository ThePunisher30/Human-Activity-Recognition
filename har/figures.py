"""Summary figures built from the saved results (no retraining).

Reads the CSV/JSON files that the experiment modules wrote into results/
and renders the remaining publication-style charts:

    tuned_leaderboard.png   -- subject-CV vs test accuracy per tuned model
    per_class_f1.png        -- per-class F1: tuned flat model vs CNN
    error_pairs.png         -- which confusion pairs cause the errors
    results_overview.png    -- one combined panel, useful as a repo banner

Run:  python -m har.figures   (after har.train / har.tune / har.deep)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .evaluate import RESULTS


def tuned_leaderboard(ax) -> None:
    df = pd.read_csv(RESULTS / "tuning.csv").sort_values("test_accuracy")
    y = range(len(df))
    ax.barh([i + 0.2 for i in y], df["test_accuracy"], 0.4,
            label="Official test (unseen subjects)")
    ax.barh([i - 0.2 for i in y], df["subject_cv"].fillna(0), 0.4,
            label="Subject-wise CV")
    ax.set_yticks(list(y), df["model"])
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_title("Tuned models (search scored with GroupKFold)")
    for i, v in zip(y, df["test_accuracy"]):
        ax.text(v + 0.002, i + 0.2, f"{v:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)


def per_class_f1(ax) -> None:
    cnn = json.loads((RESULTS / "cnn.json").read_text())
    flat = json.loads((RESULTS / "best_model.json").read_text())
    classes = list(flat["per_class_f1"])
    x = range(len(classes))
    ax.bar([i - 0.2 for i in x], flat["per_class_f1"].values(), 0.4,
           label=f"{flat['model']} (561 features)")
    ax.bar([i + 0.2 for i in x], cnn["per_class_f1"].values(), 0.4,
           label="1D CNN (raw signals)")
    ax.set_xticks(list(x), [c.replace("WALKING_", "WALK\n") for c in classes],
                  fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1: engineered features vs raw signals")
    ax.axhline(1.0, ls=":", c="gray", lw=0.7)
    ax.legend(fontsize=8)


def error_pairs(ax) -> None:
    df = pd.read_csv(RESULTS / "error_pairs.csv").head(6).iloc[::-1]
    labels = [f"{t} → {p}" for t, p in zip(df["true"], df["pred"])]
    ax.barh(labels, df["share"] * 100, color="#c44e52")
    ax.set_xlabel("Share of all test errors (%)")
    ax.set_title("Half of all errors are one confusion pair")
    for i, v in enumerate(df["share"] * 100):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    panels = [
        ("tuned_leaderboard.png", tuned_leaderboard, (8, 4.5)),
        ("per_class_f1.png", per_class_f1, (8, 4.5)),
        ("error_pairs.png", error_pairs, (8, 4)),
    ]
    for fname, draw, size in panels:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=150)
        plt.close(fig)
        print(f"wrote results/{fname}")

    # Combined 2x2 overview panel (repo banner).
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    tuned_leaderboard(axes[0, 0])
    per_class_f1(axes[0, 1])
    error_pairs(axes[1, 0])
    curve = pd.read_csv(RESULTS / "feature_selection_curve.csv")
    ax = axes[1, 1]
    ax.plot(curve["n_features"], curve["test_accuracy"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Features used (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("~200 of 561 features are enough")
    ax.grid(alpha=0.3)
    fig.suptitle("Human Activity Recognition -- results overview", y=0.995)
    fig.tight_layout()
    fig.savefig(RESULTS / "results_overview.png", dpi=150)
    plt.close(fig)
    print("wrote results/results_overview.png")


if __name__ == "__main__":
    main()
