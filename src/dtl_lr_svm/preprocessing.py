"""
Pemuatan data, pembersihan, rekayasa fitur, encoding, dan penskalaan.

Seluruh transformasi ditulis dari nol (numpy + pandas hanya untuk I/O tabel);
tidak ada ``sklearn.preprocessing`` yang dipakai pada jalur *from scratch*.
Statistik untuk penskalaan dipelajari **hanya** dari data latih lalu diterapkan
apa adanya ke data uji, sehingga tidak terjadi kebocoran informasi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TARGET = "loan_status"
ID_COL = "person_id"

BINARY_MAPS: Dict[str, Dict[str, int]] = {
    "person_gender": {"female": 0, "male": 1},
    "previous_loan_defaults_on_file": {"No": 0, "Yes": 1},
}
ONEHOT_COLS = ["person_home_ownership"]
ONEHOT_LEVELS = {"person_home_ownership": ["MORTGAGE", "OTHER", "OWN", "RENT"]}


# --------------------------------------------------------------------------- #


def load_raw(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample


# --------------------------------------------------------------------------- #
# Pembersihan dan rekayasa fitur
# --------------------------------------------------------------------------- #


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Perbaiki nilai yang tidak masuk akal secara domain.

    * ``person_age`` sampai 144 tahun dan ``person_emp_exp`` sampai 125 tahun
      jelas galat pencatatan; keduanya dipangkas ke batas wajar.
    * Lama bekerja tidak mungkin melebihi ``umur - 16``.
    * ``person_income`` sangat menceng ke kanan (maks 7,2 juta vs median 67 rb),
      sehingga dipangkas pada persentil ekstrem sebelum ditransformasi log.
    """
    df = df.copy()
    df["person_age"] = df["person_age"].clip(20, 80)
    df["person_emp_exp"] = df["person_emp_exp"].clip(0, 60)
    df["person_emp_exp"] = np.minimum(df["person_emp_exp"], df["person_age"] - 16)
    df["person_emp_exp"] = df["person_emp_exp"].clip(lower=0)
    df["person_income"] = df["person_income"].clip(4_000, 1_000_000)
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].clip(0, 30)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan yang bermakna secara kredit.

    Semuanya rasio/kombinasi yang biasa dipakai analis risiko: beban cicilan
    terhadap pendapatan, total bunga, sisa pendapatan setelah pinjaman, serta
    proksi kedewasaan finansial peminjam.
    """
    df = df.copy()
    inc = df["person_income"]
    amt = df["loan_amnt"]
    rate = df["loan_int_rate"] / 100.0

    df["log_income"] = np.log1p(inc)
    df["log_loan_amnt"] = np.log1p(amt)
    df["loan_to_income"] = amt / inc
    df["total_interest"] = amt * rate
    df["interest_to_income"] = (amt * rate) / inc
    df["income_after_loan"] = inc - amt
    df["log_income_per_year_exp"] = np.log1p(inc / (df["person_emp_exp"] + 1))
    df["age_started_work"] = df["person_age"] - df["person_emp_exp"]
    df["credit_hist_ratio"] = df["cb_person_cred_hist_length"] / df["person_age"]
    df["credit_score_x_rate"] = df["credit_score"] * df["loan_int_rate"]
    df["rate_minus_median"] = df["loan_int_rate"] - 11.01  # median suku bunga latih
    df["amt_per_credit_point"] = amt / df["credit_score"]
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding kategorik: biner untuk 2 level, one-hot untuk sisanya.

    Daftar level di-*hardcode* (``ONEHOT_LEVELS``) supaya urutan dan jumlah
    kolom hasil encoding train dan test dijamin identik.
    """
    df = df.copy()
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping).astype(float)
    for col in ONEHOT_COLS:
        for level in ONEHOT_LEVELS[col]:
            df[f"{col}={level}"] = (df[col] == level).astype(float)
        df = df.drop(columns=[col])
    return df


# --------------------------------------------------------------------------- #
# Penskalaan
# --------------------------------------------------------------------------- #


@dataclass
class StandardScaler:
    """Standardisasi z-score; kolom dengan ragam nol dibiarkan apa adanya."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1.0
        self.scale_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Pipeline lengkap
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    test_ids: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    X_train_raw: np.ndarray | None = None   # sebelum penskalaan (untuk decision tree)
    X_test_raw: np.ndarray | None = None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Dataset(train={self.X_train.shape}, test={self.X_test.shape}, "
            f"n_fitur={len(self.feature_names)}, positif={self.y_train.mean():.3f})"
        )


def build_dataset(
    data_dir: str | Path,
    use_engineered: bool = True,
    scale: bool = True,
) -> Dataset:
    """Bangun matriks fitur siap latih dari berkas CSV mentah."""
    train, test, _ = load_raw(data_dir)

    y = train[TARGET].to_numpy().astype(int)
    test_ids = test[ID_COL].to_numpy()

    tr = train.drop(columns=[TARGET, ID_COL])
    te = test.drop(columns=[ID_COL])

    tr, te = clean(tr), clean(te)
    if use_engineered:
        tr, te = engineer(tr), engineer(te)
    tr, te = encode(tr), encode(te)

    te = te[tr.columns]  # samakan urutan kolom
    feature_names = list(tr.columns)

    X_raw = tr.to_numpy(dtype=float)
    Xt_raw = te.to_numpy(dtype=float)

    if scale:
        scaler = StandardScaler().fit(X_raw)
        X = scaler.transform(X_raw)
        Xt = scaler.transform(Xt_raw)
    else:
        X, Xt = X_raw, Xt_raw

    return Dataset(
        X_train=X,
        y_train=y,
        X_test=Xt,
        test_ids=test_ids,
        feature_names=feature_names,
        X_train_raw=X_raw,
        X_test_raw=Xt_raw,
    )


# --------------------------------------------------------------------------- #
# Random Fourier Features (bonus — aproksimasi kernel RBF)
# --------------------------------------------------------------------------- #


@dataclass
class RandomFourierFeatures:
    """Aproksimasi kernel RBF dengan *Random Fourier Features*.

    Rahimi & Recht (2007) menunjukkan bahwa kernel Gaussian dapat didekati
    sebagai hasil kali dalam pada ruang berdimensi ``D``::

        z(x) = sqrt(2/D) * cos(W x + b),   W ~ N(0, 2*gamma),  b ~ U(0, 2*pi)
        k(x, y) = exp(-gamma ||x-y||^2)  ~=  z(x)^T z(y)

    Dengan begitu SVM kernel non-linier dapat dijalankan sebagai SVM **linier**
    pada ``z(x)``, sehingga biaya berubah dari O(n^2) (matriks Gram penuh, tidak
    mungkin untuk 28.800 sampel) menjadi O(n*D).

    Referensi: A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel
    Machines," NeurIPS 2007.
    """

    n_components: int = 500
    gamma: float = 0.05
    seed: int = 42
    W_: np.ndarray | None = None
    b_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RandomFourierFeatures":
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.W_ = rng.normal(0.0, np.sqrt(2 * self.gamma), size=(d, self.n_components))
        self.b_ = rng.uniform(0.0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.W_ is not None and self.b_ is not None
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W_ + self.b_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
