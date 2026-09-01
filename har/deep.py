"""1D CNN on the raw inertial signals.

Every classical model here leans on the dataset's 561 hand-engineered
features. This module asks: can a small convolutional network learn its own
features from the raw 9-channel, 128-sample windows (2.56 s at 50 Hz) and
match them? Validation subjects are held out from the training split, so
model selection is subject-wise too.

Run:  python -m har.deep [--epochs 40]
"""

import argparse
import json

import numpy as np

from . import ACTIVITY_NAMES, data
from .evaluate import RESULTS, SEED

VAL_SUBJECTS = 4  # held out from the training split for early stopping


def build_model(torch):
    nn = torch.nn

    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        def forward(self, x):
            return self.net(x)

    return nn.Sequential(
        ConvBlock(9, 32),
        ConvBlock(32, 64),
        ConvBlock(64, 64),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64, 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is required for the CNN: pip install torch"
        )
    torch.manual_seed(SEED)

    train, test = data.load(with_raw=True)

    # Hold out whole subjects for validation, never single windows.
    rng = np.random.default_rng(SEED)
    subjects = np.unique(train.subject)
    val_subjects = rng.choice(subjects, VAL_SUBJECTS, replace=False)
    val_mask = np.isin(train.subject, val_subjects)
    print(f"validation subjects: {sorted(val_subjects.tolist())}")

    def tensors(raw, y):
        return (torch.tensor(raw, dtype=torch.float32),
                torch.tensor(y - 1, dtype=torch.long))

    X_tr, y_tr = tensors(train.raw[~val_mask], train.y[~val_mask])
    X_va, y_va = tensors(train.raw[val_mask], train.y[val_mask])
    X_te, y_te = tensors(test.raw, test.y)

    model = build_model(torch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=args.batch_size, shuffle=True,
    )

    @torch.no_grad()
    def accuracy(X, y):
        model.eval()
        return (model(X).argmax(1) == y).float().mean().item()

    best_val, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        val_acc = accuracy(X_va, y_va)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  val acc {val_acc:.4f}")

    model.load_state_dict(best_state)
    test_acc = accuracy(X_te, y_te)

    model.eval()
    with torch.no_grad():
        pred = model(X_te).argmax(1).numpy() + 1
    from sklearn.metrics import f1_score
    per_class = {
        ACTIVITY_NAMES[k]: float(v)
        for k, v in zip(sorted(ACTIVITY_NAMES),
                        f1_score(test.y, pred, average=None))
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cnn.json").write_text(json.dumps({
        "val_accuracy": best_val,
        "test_accuracy": test_acc,
        "macro_f1": float(f1_score(test.y, pred, average="macro")),
        "per_class_f1": per_class,
        "input": "raw 9x128 inertial signals (no hand-engineered features)",
    }, indent=2))
    print(f"\nCNN on raw signals -- test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
