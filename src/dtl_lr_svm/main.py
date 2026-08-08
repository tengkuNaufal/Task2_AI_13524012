"""
Runner eksperimen DTL, LR, dan SVM.

Cara pakai (dijalankan dari direktori ``src/``)::

    python -m dtl_lr_svm.main --task all          # CV + banding sklearn + plot + submission
    python -m dtl_lr_svm.main --task cv           # hanya validasi silang
    python -m dtl_lr_svm.main --task submit       # hanya melatih ulang & membuat submission
    python -m dtl_lr_svm.main --task plots        # hanya membuat gambar

Keluaran disimpan ke ``results/dtl_lr_svm/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if __package__ in (None, ""):  # memungkinkan pemanggilan langsung berkas ini
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtl_lr_svm import metrics as M
from dtl_lr_svm import plots as P
from dtl_lr_svm.dtl import DecisionTreeClassifier
from dtl_lr_svm.logistic_regression import LogisticRegression
from dtl_lr_svm.preprocessing import RandomFourierFeatures, build_dataset
from dtl_lr_svm.svm import KernelSVM, LinearSVM

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "results" / "dtl_lr_svm"

# Hyperparameter terpilih setelah penelusuran (lihat notebooks/dtl_lr_svm).
BEST = {
    "dtl": dict(
        max_depth=8, min_samples_leaf=5, min_samples_split=10,
        criterion="entropy", ccp_alpha=5e-4,
    ),
    "lr": dict(optimizer="adam", lr=0.05, n_iter=60, l2=1e-4, batch_size=512),
    "svm": dict(optimizer="pegasos", lambda_=1e-3, n_iter=20, batch_size=256),
    "rff": dict(n_components=800, gamma=0.03, seed=7),
}


# --------------------------------------------------------------------------- #


def _sklearn_models():
    """Pembanding scikit-learn dengan hyperparameter sepadan."""
    from sklearn.linear_model import LogisticRegression as SkLR
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier as SkDT

    return {
        "DTL  (sklearn)": lambda: SkDT(
            criterion="entropy", max_depth=8, min_samples_leaf=5,
            min_samples_split=10, ccp_alpha=5e-4, random_state=42,
        ),
        "LR   (sklearn)": lambda: SkLR(C=1 / (1e-4 * 28800), max_iter=3000),
        "SVM  (sklearn)": lambda: LinearSVC(C=1 / (1e-3 * 28800), max_iter=8000, dual="auto"),
    }


def run_cv(ds, n_splits: int, seed: int, with_rff: bool = True) -> Dict[str, dict]:
    """Validasi silang untuk seluruh model, from scratch maupun sklearn."""
    X, Xr, y = ds.X_train, ds.X_train_raw, ds.y_train
    results: Dict[str, dict] = {}

    jobs: List[Tuple[str, object, np.ndarray]] = [
        ("DTL  (from scratch)", lambda: DecisionTreeClassifier(**BEST["dtl"]), Xr),
        ("LR   (from scratch)", lambda: LogisticRegression(**BEST["lr"]), X),
        ("SVM  (from scratch)", lambda: LinearSVM(**BEST["svm"]), X),
    ]

    if with_rff:
        Z = RandomFourierFeatures(**BEST["rff"]).fit_transform(X)
        jobs.append(("LR   (RFF, from scratch)", lambda: LogisticRegression(**BEST["lr"]), Z))
        jobs.append(("SVM  (RFF, from scratch)", lambda: LinearSVM(**BEST["svm"]), Z))

    for name, factory, mat in jobs:
        t0 = time.perf_counter()
        r = M.cross_validate(factory, mat, y, n_splits=n_splits, seed=seed)
        r["duration_s"] = round(time.perf_counter() - t0, 2)
        results[name] = r
        print(
            f"  {name:<26} acc={r['accuracy_mean']:.4f}+-{r['accuracy_std']:.4f} "
            f"f1={r['f1_mean']:.4f} auc={r['roc_auc_mean']:.4f}  ({r['duration_s']}s)"
        )

    for name, factory in _sklearn_models().items():
        mat = Xr if name.startswith("DTL") else X
        t0 = time.perf_counter()
        r = M.cross_validate(factory, mat, y, n_splits=n_splits, seed=seed)
        r["duration_s"] = round(time.perf_counter() - t0, 2)
        results[name] = r
        print(
            f"  {name:<26} acc={r['accuracy_mean']:.4f}+-{r['accuracy_std']:.4f} "
            f"f1={r['f1_mean']:.4f} auc={r['roc_auc_mean']:.4f}  ({r['duration_s']}s)"
        )

    return results


def run_smo_demo(ds, n: int = 2500, seed: int = 42) -> dict:
    """Verifikasi implementasi SVM dual (SMO) pada subsampel."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds.y_train), size=n, replace=False)
    split = int(0.75 * n)
    tr, va = idx[:split], idx[split:]

    out = {}
    for kernel, gamma in [("linear", 0.0), ("rbf", 0.03)]:
        t0 = time.perf_counter()
        m = KernelSVM(C=1.0, kernel=kernel, gamma=gamma, max_iter=60).fit(
            ds.X_train[tr], ds.y_train[tr]
        )
        pred = m.predict(ds.X_train[va])
        rep = M.classification_report(
            ds.y_train[va], pred, m.decision_function(ds.X_train[va])
        )
        rep["n_support_vectors"] = m.n_support_vectors
        rep["duration_s"] = round(time.perf_counter() - t0, 2)
        out[f"KernelSVM-SMO ({kernel})"] = rep
        print(f"  {f'SMO {kernel}':<26} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))
    return out


# --------------------------------------------------------------------------- #


def make_plots(ds, cv: Dict[str, dict], out: Path) -> None:
    X, Xr, y = ds.X_train, ds.X_train_raw, ds.y_train
    names = ds.feature_names

    # 1. pohon keputusan + importance
    tree = DecisionTreeClassifier(**BEST["dtl"]).fit(Xr, y, names)
    (out / "tree_structure.txt").write_text(tree.export_text(max_depth=4), encoding="utf-8")
    print("  ->", P.plot_tree(tree, out / "decision_tree.png", max_depth=3, feature_names=names))
    print("  ->", P.plot_feature_importance(tree.feature_importances(), names, out / "feature_importance.png"))

    # 2. kurva galat tiap optimizer
    lr_curves = {}
    for opt, lr_, it in [("batch", 0.5, 120), ("sgd", 0.05, 60), ("adam", 0.05, 60)]:
        m = LogisticRegression(optimizer=opt, lr=lr_, n_iter=it, l2=1e-4).fit(X, y)
        lr_curves[f"{opt} (lr={lr_})"] = m.loss_history
    print("  ->", P.plot_loss_curves(lr_curves, out / "lr_loss_curves.png",
                                     "Logistic Regression — galat cross-entropy per epoch"))

    svm_curves = {}
    for opt, lam in [("pegasos", 1e-3), ("subgradient", 1e-4)]:
        m = LinearSVM(optimizer=opt, lambda_=lam, n_iter=30, lr=0.03).fit(X, y)
        svm_curves[f"{opt} (lambda={lam})"] = m.loss_history
    print("  ->", P.plot_loss_curves(svm_curves, out / "svm_loss_curves.png",
                                     "Linear SVM — nilai fungsi objektif hinge per epoch",
                                     ylabel="hinge + L2"))

    # 3. kontur galat LR pada dua fitur terkuat + lintasan optimizer
    f1 = names.index("loan_percent_income")
    f2 = names.index("loan_int_rate")
    X2 = X[:, [f1, f2]]
    # intercept dimatikan agar ruang parameter benar-benar dua dimensi,
    # sehingga kontur yang digambar persis fungsi galat yang dioptimasi
    paths = {}
    for opt, lr_, it in [("batch GD", 0.5, 150), ("mini-batch SGD", 0.05, 60), ("Adam", 0.05, 60)]:
        key = {"batch GD": "batch", "mini-batch SGD": "sgd", "Adam": "adam"}[opt]
        m = LogisticRegression(
            optimizer=key, lr=lr_, n_iter=it, l2=1e-4, fit_intercept=False, tol=0.0
        )
        m.fit(X2, y, track_path=True)
        paths[f"{opt} (lr={lr_})"] = np.array([p[1:] for p in m.weight_path])
    print("  ->", P.plot_lr_contour(X2, y.astype(float), paths, out / "lr_loss_contour.png",
                                    ("loan_percent_income", "loan_int_rate")))

    # 4. batas keputusan ketiga model pada dua fitur yang sama
    models2 = {
        "DTL (CART)": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20).fit(X2, y),
        "Logistic Regression": LogisticRegression(**BEST["lr"]).fit(X2, y),
        "Linear SVM (Pegasos)": LinearSVM(**BEST["svm"]).fit(X2, y),
    }
    print("  ->", P.plot_decision_boundary(models2, X2, y, out / "decision_boundaries.png",
                                           ("loan_percent_income", "loan_int_rate")))

    # 5. ROC + matriks konfusi dari skor out-of-fold
    roc = {}
    cms = {}
    for name in ("DTL  (from scratch)", "LR   (from scratch)", "SVM  (from scratch)"):
        if name not in cv:
            continue
        s = np.asarray(cv[name]["oof_score"])
        roc[name.strip()] = (y, s)
        thr = 0.5 if "SVM" not in name else 0.0
        cms[name.strip()] = M.confusion_matrix(y, (s >= thr).astype(int))
    if roc:
        print("  ->", P.plot_roc(roc, out / "roc_curves.png"))
        print("  ->", P.plot_confusion_matrices(cms, out / "confusion_matrices.png"))

    # 6. batang perbandingan from scratch vs sklearn
    rows = [
        (n.replace("(from scratch)", "(scratch)"), r["accuracy_mean"], r["accuracy_std"])
        for n, r in cv.items()
    ]
    if rows:
        print("  ->", P.plot_cv_comparison(rows, out / "cv_comparison.png"))


# --------------------------------------------------------------------------- #


def make_submission(ds, out: Path, model_key: str = "dtl") -> Path:
    """Latih ulang model terpilih pada seluruh data latih lalu prediksi test.csv."""
    y = ds.y_train
    if model_key == "dtl":
        model = DecisionTreeClassifier(**BEST["dtl"]).fit(ds.X_train_raw, y, ds.feature_names)
        pred = model.predict(ds.X_test_raw)
    elif model_key == "lr":
        model = LogisticRegression(**BEST["lr"]).fit(ds.X_train, y)
        pred = model.predict(ds.X_test)
    elif model_key == "svm":
        model = LinearSVM(**BEST["svm"]).fit(ds.X_train, y)
        pred = model.predict(ds.X_test)
    else:  # pragma: no cover
        raise ValueError(model_key)

    import pandas as pd

    sub = pd.DataFrame({"person_id": ds.test_ids, "loan_status": pred.astype(int)})
    path = out / f"submission_{model_key}.csv"
    sub.to_csv(path, index=False)
    print(
        f"  submission {model_key}: {path}  "
        f"(prediksi positif {sub.loan_status.mean():.3%}, latih {y.mean():.3%})"
    )
    return path


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dtl_lr_svm",
        description="Eksperimen DTL / Logistic Regression / SVM from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", default="all", choices=["all", "cv", "submit", "plots", "smo"])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-rff", action="store_true", help="lewati varian Random Fourier Features")
    p.add_argument("--submit-model", default="dtl", choices=["dtl", "lr", "svm"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DTL, LOGISTIC REGRESSION, DAN SVM — IMPLEMENTASI FROM SCRATCH")
    print("=" * 78)
    t0 = time.perf_counter()
    ds = build_dataset(args.data)
    print(f"Dataset          : {ds}")
    print(f"Jumlah fitur     : {len(ds.feature_names)}")
    print(f"Proporsi positif : {ds.y_train.mean():.4f}")
    print(f"Waktu pramuat    : {time.perf_counter() - t0:.2f}s")
    print()

    cv: Dict[str, dict] = {}
    payload: Dict[str, object] = {
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.test_ids)),
        "features": ds.feature_names,
        "hyperparameters": BEST,
        "folds": args.folds,
        "seed": args.seed,
    }

    if args.task in ("all", "cv", "plots"):
        print("-" * 78)
        print(f"VALIDASI SILANG {args.folds}-FOLD (stratified)")
        print("-" * 78)
        cv = run_cv(ds, args.folds, args.seed, with_rff=not args.no_rff)
        print()
        payload["cv"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "oof_score"}
            for k, v in cv.items()
        }

    if args.task in ("all", "smo"):
        print("-" * 78)
        print("VERIFIKASI SVM DUAL (SMO) PADA SUBSAMPEL 2.500 BARIS")
        print("-" * 78)
        payload["smo"] = run_smo_demo(ds)
        print()

    if args.task in ("all", "plots"):
        print("-" * 78)
        print("MEMBUAT GAMBAR")
        print("-" * 78)
        make_plots(ds, cv, out)
        print()

    if args.task in ("all", "submit"):
        print("-" * 78)
        print("SUBMISSION KAGGLE")
        print("-" * 78)
        for key in ("dtl", "lr", "svm"):
            make_submission(ds, out, key)
        print()

    if cv:
        print("=" * 78)
        print("RINGKASAN PERBANDINGAN")
        print("=" * 78)
        print(f"{'Model':<28} {'akurasi':>16} {'F1':>8} {'ROC-AUC':>9} {'detik':>7}")
        print("-" * 72)
        for name, r in sorted(cv.items(), key=lambda kv: -kv[1]["accuracy_mean"]):
            print(
                f"{name:<28} {r['accuracy_mean']:.4f}+-{r['accuracy_std']:.4f} "
                f"{r['f1_mean']:>8.4f} {r['roc_auc_mean']:>9.4f} {r['duration_s']:>7.1f}"
            )
        print()

    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"Ringkasan JSON -> {out / 'results.json'}")
    print(f"Total waktu    : {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
