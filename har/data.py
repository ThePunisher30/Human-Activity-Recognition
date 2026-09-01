"""Download and load the UCI HAR dataset."""

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DATASET_DIR = "UCI HAR Dataset"

RAW_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


@dataclass
class Split:
    """One split (train or test) of the dataset."""

    X: np.ndarray          # (n, 561) engineered features
    y: np.ndarray          # (n,) activity labels 1..6
    subject: np.ndarray    # (n,) subject ids 1..30
    raw: np.ndarray | None = None  # (n, 9, 128) inertial signals


def download(data_dir: str | Path = "data") -> Path:
    """Fetch and extract the dataset if not already present."""
    data_dir = Path(data_dir)
    target = data_dir / DATASET_DIR
    if target.exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    # The UCI archive nests the real dataset zip inside the download.
    with outer.open(f"{DATASET_DIR}.zip") as inner:
        zipfile.ZipFile(io.BytesIO(inner.read())).extractall(data_dir)
    return target


def feature_names(data_dir: str | Path = "data") -> list[str]:
    path = Path(data_dir) / DATASET_DIR / "features.txt"
    return [line.split(maxsplit=1)[1].strip() for line in path.open()]


def load_split(split: str, data_dir: str | Path = "data",
               with_raw: bool = False) -> Split:
    """Load 'train' or 'test'. Set with_raw for the (n, 9, 128) signals."""
    base = Path(data_dir) / DATASET_DIR / split
    X = np.loadtxt(base / f"X_{split}.txt")
    y = np.loadtxt(base / f"y_{split}.txt", dtype=int)
    subject = np.loadtxt(base / f"subject_{split}.txt", dtype=int)
    raw = None
    if with_raw:
        sig_dir = base / "Inertial Signals"
        raw = np.stack(
            [np.loadtxt(sig_dir / f"{name}_{split}.txt") for name in RAW_SIGNALS],
            axis=1,
        )
    return Split(X=X, y=y, subject=subject, raw=raw)


def load(data_dir: str | Path = "data",
         with_raw: bool = False) -> tuple[Split, Split]:
    download(data_dir)
    return (load_split("train", data_dir, with_raw),
            load_split("test", data_dir, with_raw))
