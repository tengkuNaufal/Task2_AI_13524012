"""
Metrik evaluasi dan validasi silang — seluruhnya ditulis dari nol dengan numpy.

Dipakai baik oleh implementasi *from scratch* maupun saat membandingkannya
dengan scikit-learn, sehingga angka yang dibandingkan benar-benar dihitung
dengan cara yang sama.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Metrik klasifikasi biner
# --------------------------------------------------------------------------- #


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Matriks konfusi 2x2 dengan tata letak ``[[TN, FP], [FN, TP]]``."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = int(np.sum((y_true == t) & (y_pred == p)))
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if p + r else 0.0


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC lewat statistik peringkat Mann-Whitney U (tahan nilai kembar)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # rata-rata peringkat untuk nilai kembar
        i = j + 1
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None
) -> Dict[str, float]:
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if score is not None:
        out["roc_auc"] = roc_auc(y_true, score)
    return out


def format_report(name: str, rep: Dict[str, float]) -> str:
    parts = " ".join(f"{k}={v:.4f}" for k, v in rep.items())
    return f"{name:<28} {parts}"


# --------------------------------------------------------------------------- #
# Validasi silang
# --------------------------------------------------------------------------- #


def stratified_kfold(
    y: np.ndarray, n_splits: int = 5, shuffle: bool = True, seed: int = 42
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-Fold: proporsi tiap kelas dijaga sama di setiap lipatan."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds: List[List[np.ndarray]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if shuffle:
            rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_splits)):
            folds[k].append(chunk)
    fold_idx = [np.concatenate(f) for f in folds]
    all_idx = np.arange(len(y))
    for k in range(n_splits):
        val = np.sort(fold_idx[k])
        train = np.setdiff1d(all_idx, val, assume_unique=False)
        yield train, val


def cross_validate(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    threshold: float = 0.5,
    verbose: bool = False,
) -> Dict[str, object]:
    """Jalankan K-Fold CV; ``model_factory()`` harus mengembalikan model baru.

    Model wajib punya ``fit(X, y)``, ``predict(X)``, dan (opsional)
    ``decision_function``/``predict_proba`` untuk perhitungan ROC-AUC.
    """
    accs, f1s, aucs, precs, recs = [], [], [], [], []
    oof = np.zeros(len(y), dtype=float)

    for k, (tr, va) in enumerate(stratified_kfold(y, n_splits, seed=seed), start=1):
        model = model_factory()
        model.fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            s = model.predict_proba(X[va])
            s = s[:, 1] if getattr(s, "ndim", 1) == 2 else s
        elif hasattr(model, "decision_function"):
            s = model.decision_function(X[va])
        else:  # pragma: no cover
            s = model.predict(X[va]).astype(float)
        oof[va] = s
        pred = model.predict(X[va])
        accs.append(accuracy(y[va], pred))
        f1s.append(f1_score(y[va], pred))
        precs.append(precision(y[va], pred))
        recs.append(recall(y[va], pred))
        aucs.append(roc_auc(y[va], s))
        if verbose:
            print(
                f"   fold {k}: acc={accs[-1]:.4f} f1={f1s[-1]:.4f} auc={aucs[-1]:.4f}"
            )

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "fold_accuracy": accs,
        "fold_f1": f1s,
        "oof_score": oof,
    }


def best_threshold(y_true: np.ndarray, score: np.ndarray, metric=f1_score) -> Tuple[float, float]:
    """Cari ambang keputusan terbaik pada skor out-of-fold."""
    cands = np.quantile(score, np.linspace(0.01, 0.99, 199))
    best_t, best_v = 0.5, -np.inf
    for t in cands:
        v = metric(y_true, (score >= t).astype(int))
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v
