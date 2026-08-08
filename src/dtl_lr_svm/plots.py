"""
Visualisasi untuk bagian DTL, LR, dan SVM.

Berisi penggambar pohon keputusan hasil implementasi sendiri, kurva galat
pelatihan, kontur fungsi galat Logistic Regression beserta lintasan parameter,
kurva ROC, dan matriks konfusi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Pohon keputusan
# --------------------------------------------------------------------------- #


def _layout(node, depth: int, max_depth: int, counter: List[float]) -> Dict:
    """Tata letak pohon: daun diberi absis berurutan, simpul dalam di tengah anaknya."""
    if node.is_leaf or depth >= max_depth:
        x = counter[0]
        counter[0] += 1.0
        return {"node": node, "x": x, "y": -depth, "children": [], "cut": not node.is_leaf}
    left = _layout(node.left, depth + 1, max_depth, counter)
    right = _layout(node.right, depth + 1, max_depth, counter)
    return {
        "node": node,
        "x": 0.5 * (left["x"] + right["x"]),
        "y": -depth,
        "children": [left, right],
        "cut": False,
    }


def plot_tree(
    tree,
    path: str | Path,
    max_depth: int = 3,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Decision Tree (CART) hasil implementasi from scratch",
) -> Path:
    """Gambar percabangan pohon sampai kedalaman ``max_depth``.

    Warna simpul menyatakan proporsi kelas positif: makin merah makin besar
    peluang ``loan_status = 1``.
    """
    plt = _plt()
    names = list(feature_names or tree.feature_names_ or
                 [f"x[{i}]" for i in range(tree.n_features_)])

    counter = [0.0]
    root = _layout(tree.root, 0, max_depth, counter)
    n_leaves_drawn = counter[0]

    fig, ax = plt.subplots(figsize=(max(11.0, n_leaves_drawn * 1.9), 2.4 * (max_depth + 1)))
    cmap = plt.get_cmap("RdYlGn_r")

    def draw(item, parent=None, side: str = "") -> None:
        node = item["node"]
        x, y = item["x"], item["y"]
        if parent is not None:
            ax.plot([parent["x"], x], [parent["y"] - 0.22, y + 0.22],
                    color="0.5", lw=1.0, zorder=1)
            ax.text(
                0.5 * (parent["x"] + x), 0.5 * (parent["y"] + y) + 0.02, side,
                fontsize=7, ha="center", va="center", color="0.25",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
            )

        if item["children"]:
            label = (
                f"{names[node.feature]}\n<= {node.threshold:.4g}\n"
                f"n={node.n_samples}  imp={node.impurity:.3f}"
            )
        else:
            kind = "(dipotong)" if item["cut"] else "LEAF"
            label = f"{kind}\nn={node.n_samples}\np(1)={node.proba:.3f}\nkelas {node.prediction}"

        ax.text(
            x, y, label, fontsize=7.5, ha="center", va="center", zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc=cmap(node.proba), ec="0.3", lw=0.8, alpha=0.9,
            ),
        )
        for k, child in enumerate(item["children"]):
            draw(child, item, "ya" if k == 0 else "tidak")

    draw(root)
    ax.set_xlim(-0.8, max(n_leaves_drawn - 0.2, 1))
    ax.set_ylim(-max_depth - 0.6, 0.6)
    ax.axis("off")
    ax.set_title(f"{title}\n(ditampilkan sampai kedalaman {max_depth})", fontsize=11)
    return _save(fig, path)


def plot_feature_importance(
    importances: np.ndarray, names: Sequence[str], path: str | Path, top: int = 15
) -> Path:
    plt = _plt()
    order = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(order) + 1.4))
    ax.barh([names[i] for i in order], importances[order], color="teal")
    ax.set_xlabel("Gini/entropy importance (ternormalisasi)")
    ax.set_title(f"{top} fitur terpenting menurut Decision Tree from scratch")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Logistic Regression
# --------------------------------------------------------------------------- #


def plot_loss_curves(
    curves: Dict[str, Sequence[float]], path: str | Path, title: str, ylabel: str = "Galat"
) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, hist in curves.items():
        ax.plot(range(1, len(hist) + 1), hist, label=name, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_lr_contour(
    X2: np.ndarray,
    y: np.ndarray,
    paths: Dict[str, np.ndarray],
    path: str | Path,
    feature_names: Tuple[str, str] = ("x1", "x2"),
    l2: float = 1e-4,
    grid: int = 90,
) -> Path:
    """Kontur galat cross-entropy pada ruang dua bobot beserta lintasan optimizer.

    Supaya konturnya **benar-benar** galat model (bukan proyeksi/aproksimasi),
    Logistic Regression dilatih ulang hanya dengan dua fitur sehingga ruang
    parameternya memang berdimensi dua (intercept difiksasi pada nilai
    optimumnya). ``paths[name]`` berisi larik ``(n_epoch, 2)`` berisi lintasan
    ``(w1, w2)`` tiap optimizer.
    """
    plt = _plt()
    from .logistic_regression import sigmoid

    all_pts = np.vstack(list(paths.values()))
    pad = 0.6 * max(1.0, np.ptp(all_pts, axis=0).max())
    w1 = np.linspace(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad, grid)
    w2 = np.linspace(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad, grid)
    W1, W2 = np.meshgrid(w1, w2)

    Z = np.zeros_like(W1)
    for i in range(grid):
        z = X2 @ np.vstack([W1[i], W2[i]])            # (n, grid)
        p = np.clip(sigmoid(z), 1e-12, 1 - 1e-12)
        ce = -(y[:, None] * np.log(p) + (1 - y)[:, None] * np.log(1 - p)).mean(axis=0)
        Z[i] = ce + 0.5 * l2 * (W1[i] ** 2 + W2[i] ** 2)

    fig, ax = plt.subplots(figsize=(8, 6.2))
    cs = ax.contourf(W1, W2, Z, levels=35, cmap="viridis")
    ax.contour(W1, W2, Z, levels=18, colors="white", linewidths=0.4, alpha=0.6)
    fig.colorbar(cs, ax=ax, label="cross-entropy + L2")

    markers = ["o", "s", "^", "D"]
    for k, (name, p) in enumerate(paths.items()):
        (line,) = ax.plot(p[:, 0], p[:, 1], "-", lw=1.8, label=name)
        c = line.get_color()
        ax.plot(p[0, 0], p[0, 1], markers[k % 4], ms=8, mfc="none", mec=c, mew=2.0)
        ax.plot(p[-1, 0], p[-1, 1], markers[k % 4], ms=9, mfc=c, mec="black", mew=0.8)

    ax.set_xlabel(f"w[{feature_names[0]}]")
    ax.set_ylabel(f"w[{feature_names[1]}]")
    ax.set_title(
        "Kontur fungsi galat Logistic Regression dan lintasan parameter\n"
        "(lingkaran kosong = titik awal, penuh = titik akhir)"
    )
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, path)


def plot_decision_boundary(
    models: Dict[str, object],
    X2: np.ndarray,
    y: np.ndarray,
    path: str | Path,
    feature_names: Tuple[str, str] = ("x1", "x2"),
    sample: int = 3000,
    seed: int = 0,
) -> Path:
    """Batas keputusan ketiga model pada dua fitur paling informatif."""
    plt = _plt()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(sample, len(y)), replace=False)

    x1 = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 220)
    x2 = np.linspace(X2[:, 1].min(), X2[:, 1].max(), 220)
    G1, G2 = np.meshgrid(x1, x2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, models.items()):
        if hasattr(m, "predict_proba"):
            Zg = np.asarray(m.predict_proba(grid), dtype=float)
        else:
            Zg = np.asarray(m.decision_function(grid), dtype=float)
        ax.contourf(G1, G2, Zg.reshape(G1.shape), levels=25, cmap="RdYlGn_r", alpha=0.8)
        ax.contour(G1, G2, Zg.reshape(G1.shape),
                   levels=[0.5 if hasattr(m, "predict_proba") else 0.0],
                   colors="black", linewidths=1.4)
        ax.scatter(X2[idx, 0], X2[idx, 1], c=y[idx], cmap="coolwarm",
                   s=4, alpha=0.35, edgecolors="none")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    fig.suptitle("Batas keputusan ketiga model pada dua fitur terkuat", y=1.02)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Evaluasi
# --------------------------------------------------------------------------- #


def plot_roc(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], path: str | Path) -> Path:
    """``curves[name] = (y_true, score)``."""
    plt = _plt()
    from .metrics import roc_auc

    fig, ax = plt.subplots(figsize=(6.4, 6))
    for name, (y_true, score) in curves.items():
        order = np.argsort(-np.asarray(score))
        yt = np.asarray(y_true)[order]
        tpr = np.cumsum(yt) / max(1, np.sum(yt))
        fpr = np.cumsum(1 - yt) / max(1, np.sum(1 - yt))
        ax.plot(
            np.concatenate([[0], fpr]), np.concatenate([[0], tpr]),
            lw=1.6, label=f"{name} (AUC={roc_auc(y_true, score):.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC (skor out-of-fold, 5-fold CV)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def plot_confusion_matrices(
    mats: Dict[str, np.ndarray], path: str | Path
) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, len(mats), figsize=(3.7 * len(mats), 3.6))
    if len(mats) == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, mats.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=10)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["asli 0", "asli 1"])
        ax.set_title(name, fontsize=10)
    fig.suptitle("Matriks konfusi out-of-fold")
    return _save(fig, path)


def plot_cv_comparison(rows: Sequence[Tuple[str, float, float]], path: str | Path) -> Path:
    """``rows = [(nama, mean_acc, std_acc), ...]``."""
    plt = _plt()
    names = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = ["tab:blue" if "scratch" in n else "tab:orange" for n in names]

    fig, ax = plt.subplots(figsize=(9, 0.46 * len(rows) + 1.8))
    ax.barh(names, means, xerr=stds, color=colors, capsize=3)
    ax.set_xlim(min(means) - 0.03, max(means) + 0.015)
    ax.set_xlabel("Akurasi rata-rata 5-fold CV")
    ax.set_title("From scratch (biru) vs scikit-learn (oranye)")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + 0.001, i, f"{m:.4f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)
