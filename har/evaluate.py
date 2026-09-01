"""Shared evaluation helpers: CV protocols, metric tables, plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from . import ACTIVITY_NAMES

RESULTS = Path("results")
SEED = 42


def random_cv(model, X, y, folds: int = 5) -> np.ndarray:
    """Accuracy per fold with subjects mixed across folds (leaky)."""
    cv = KFold(folds, shuffle=True, random_state=SEED)
    return cross_val_score(clone(model), X, y, cv=cv, n_jobs=-1)


def subject_cv(model, X, y, groups, folds: int = 5) -> np.ndarray:
    """Accuracy per fold with every subject confined to one fold."""
    cv = GroupKFold(folds)
    return cross_val_score(clone(model), X, y, cv=cv, groups=groups, n_jobs=-1)


def test_metrics(model, X, y, X_test, y_test) -> dict:
    model = clone(model).fit(X, y)
    pred = model.predict(X_test)
    return {
        "accuracy": float((pred == y_test).mean()),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "per_class_f1": {
            ACTIVITY_NAMES[k]: float(v)
            for k, v in zip(sorted(ACTIVITY_NAMES), f1_score(y_test, pred, average=None))
        },
        "pred": pred,
    }


def plot_confusion(y_true, y_pred, path: Path, title: str) -> None:
    labels = [ACTIVITY_NAMES[k] for k in sorted(ACTIVITY_NAMES)]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
