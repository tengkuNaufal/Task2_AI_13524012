"""
Decision Tree Learning — varian **CART** (Classification and Regression Trees),
ditulis dari nol dengan numpy.

Alasan memilih CART dibanding ID3/C4.5
--------------------------------------
1. Sebelas dari tiga belas atribut dataset ini **kontinu** (pendapatan, suku
   bunga, skor kredit, ...). ID3 hanya menangani atribut kategorik sehingga
   memerlukan diskretisasi manual yang membuang informasi urutan.
2. CART memakai *binary split* ``x_j <= s`` sehingga ambang optimum dicari
   langsung pada nilai kontinu. C4.5 juga mendukung atribut kontinu, tetapi
   *gain ratio*-nya dirancang untuk mengoreksi bias atribut ber-kardinalitas
   tinggi — masalah yang praktis tidak muncul di sini karena setelah encoding
   hanya tersisa atribut numerik dan biner.
3. Impuritas **Gini** lebih murah daripada entropi (tanpa ``log``) dan pada
   praktiknya menghasilkan pohon yang setara; ini penting karena pencarian
   split dilakukan pada 28.800 baris x 24 fitur.
4. Struktur biner CART memudahkan *cost-complexity pruning* (Breiman dkk.,
   1984) yang dipakai sebagai kendali overfitting.

Kriteria ``entropy`` tetap disediakan sebagai pembanding empiris.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #


@dataclass
class Node:
    """Simpul pohon. Simpul daun ditandai ``feature is None``."""

    n_samples: int
    value: np.ndarray               # distribusi kelas [n0, n1]
    impurity: float
    depth: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    @property
    def proba(self) -> float:
        total = self.value.sum()
        return float(self.value[1] / total) if total else 0.0

    @property
    def prediction(self) -> int:
        return int(np.argmax(self.value))


# --------------------------------------------------------------------------- #


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


class DecisionTreeClassifier:
    """CART untuk klasifikasi biner.

    Parameter
    ---------
    max_depth
        Kedalaman maksimum pohon.
    min_samples_split
        Jumlah sampel minimum agar sebuah simpul boleh dipecah.
    min_samples_leaf
        Jumlah sampel minimum pada setiap anak hasil pemecahan.
    min_impurity_decrease
        Penurunan impuritas berbobot minimum agar pemecahan diterima.
    criterion
        ``"gini"`` (default) atau ``"entropy"``.
    class_weight
        ``None`` atau ``"balanced"`` — membobot kelas berbanding terbalik
        dengan frekuensinya, berguna karena kelas positif hanya 22%.
    ccp_alpha
        Parameter *cost-complexity pruning*; 0 berarti tanpa pemangkasan.
    max_features
        Bila diisi, hanya sebagian fitur acak yang dipertimbangkan tiap split.
    """

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        min_impurity_decrease: float = 0.0,
        criterion: str = "gini",
        class_weight: Optional[str] = None,
        ccp_alpha: float = 0.0,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.random_state = random_state

        self.root: Optional[Node] = None
        self.n_features_: int = 0
        self.feature_names_: Optional[List[str]] = None
        self._w = np.ones(2)
        self._rng = np.random.default_rng(random_state)

    # -- impuritas -------------------------------------------------------- #

    def _impurity(self, counts: np.ndarray) -> float:
        return _gini(counts) if self.criterion == "gini" else _entropy(counts)

    def _impurity_vec(self, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
        """Impuritas untuk banyak kandidat split sekaligus (vektorisasi)."""
        total = c0 + c1
        with np.errstate(divide="ignore", invalid="ignore"):
            p0 = np.where(total > 0, c0 / total, 0.0)
            p1 = np.where(total > 0, c1 / total, 0.0)
            if self.criterion == "gini":
                return 1.0 - p0 * p0 - p1 * p1
            t0 = np.where(p0 > 0, -p0 * np.log2(np.where(p0 > 0, p0, 1)), 0.0)
            t1 = np.where(p1 > 0, -p1 * np.log2(np.where(p1 > 0, p1, 1)), 0.0)
            return t0 + t1

    # -- pencarian split --------------------------------------------------- #

    def _best_split(
        self, X: np.ndarray, w0: np.ndarray, w1: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Cari (fitur, ambang) dengan penurunan impuritas berbobot terbesar.

        Untuk tiap fitur, nilai diurutkan sekali lalu seluruh kandidat ambang
        dievaluasi serentak memakai jumlah kumulatif — sehingga kompleksitasnya
        O(d * n log n) per simpul, bukan O(d * n^2).

        ``w0``/``w1`` adalah bobot tiap sampel untuk kelas 0 dan kelas 1
        (bernilai 0 bila sampel bukan milik kelas tersebut), agar
        ``class_weight="balanced"`` ditangani tanpa menduplikasi baris.
        """
        n, d = X.shape
        tot0, tot1 = w0.sum(), w1.sum()
        parent_imp = self._impurity_vec(np.array([tot0]), np.array([tot1]))[0]
        total_w = tot0 + tot1

        features = np.arange(d)
        if self.max_features is not None and self.max_features < d:
            features = self._rng.choice(d, size=self.max_features, replace=False)

        best_gain = self.min_impurity_decrease
        best_feat: Optional[int] = None
        best_thr: Optional[float] = None

        for f in features:
            col = X[:, f]
            order = np.argsort(col, kind="stable")
            xs = col[order]
            c1 = np.cumsum(w1[order])[:-1]
            c0 = np.cumsum(w0[order])[:-1]
            n_left = np.arange(1, n)
            n_right = n - n_left

            valid = (xs[:-1] != xs[1:]) & (n_left >= self.min_samples_leaf) & (
                n_right >= self.min_samples_leaf
            )
            if not valid.any():
                continue

            r0 = tot0 - c0
            r1 = tot1 - c1
            wl = c0 + c1
            wr = r0 + r1
            imp_l = self._impurity_vec(c0, c1)
            imp_r = self._impurity_vec(r0, r1)
            weighted = (wl * imp_l + wr * imp_r) / total_w
            gain = parent_imp - weighted
            gain = np.where(valid, gain, -np.inf)

            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain = float(gain[k])
                best_feat = int(f)
                best_thr = float(0.5 * (xs[k] + xs[k + 1]))

        return best_feat, best_thr, best_gain

    # -- pembangunan pohon ------------------------------------------------- #

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        counts = np.array(
            [self._w[0] * np.sum(y == 0), self._w[1] * np.sum(y == 1)], dtype=float
        )
        node = Node(
            n_samples=len(y),
            value=counts,
            impurity=self._impurity(counts),
            depth=depth,
        )

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or counts[0] == 0
            or counts[1] == 0
        ):
            return node

        w0 = np.where(y == 0, self._w[0], 0.0)
        w1 = np.where(y == 1, self._w[1], 0.0)
        feat, thr, _gain = self._best_split(X, w0, w1)
        if feat is None:
            return node

        mask = X[:, feat] <= thr
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node

        node.feature = feat
        node.threshold = thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.n_features_ = X.shape[1]
        self.feature_names_ = feature_names
        self._n_total = float(len(y))

        if self.class_weight == "balanced":
            n0, n1 = np.sum(y == 0), np.sum(y == 1)
            self._w = np.array([len(y) / (2 * n0), len(y) / (2 * n1)])
        else:
            self._w = np.ones(2)

        self.root = self._build(X, y, depth=0)
        if self.ccp_alpha > 0:
            self._prune(self.root)
        return self

    # -- cost-complexity pruning ------------------------------------------ #

    def _subtree_cost(self, node: Node) -> Tuple[float, int]:
        """(risiko total daun subpohon, jumlah daun).

        Risiko dinormalisasi terhadap jumlah sampel latih, ``R(t) = (n_t/N) * i(t)``,
        mengikuti konvensi scikit-learn agar besaran ``ccp_alpha`` sebanding.
        """
        if node.is_leaf:
            return node.impurity * node.value.sum() / self._n_total, 1
        cl, nl = self._subtree_cost(node.left)  # type: ignore[arg-type]
        cr, nr = self._subtree_cost(node.right)  # type: ignore[arg-type]
        return cl + cr, nl + nr

    def _prune(self, node: Node) -> None:
        """Pemangkasan *weakest-link* rekursif (Breiman dkk., 1984, Bab 3).

        Sebuah simpul dijadikan daun bila biaya kompleksitasnya sebagai daun
        tunggal, ``R(t) + alpha``, tidak lebih besar daripada biaya subpohonnya,
        ``R(T_t) + alpha*|T_t|``.
        """
        if node.is_leaf:
            return
        self._prune(node.left)   # type: ignore[arg-type]
        self._prune(node.right)  # type: ignore[arg-type]
        sub_cost, n_leaves = self._subtree_cost(node)
        leaf_cost = node.impurity * node.value.sum() / self._n_total
        if leaf_cost + self.ccp_alpha <= sub_cost + self.ccp_alpha * n_leaves:
            node.feature = None
            node.threshold = None
            node.left = None
            node.right = None

    # -- inferensi --------------------------------------------------------- #

    def _walk(self, x: np.ndarray) -> Node:
        node = self.root
        assert node is not None
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right  # type: ignore[index]
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._walk(x).proba for x in X])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # -- introspeksi ------------------------------------------------------- #

    @property
    def n_leaves(self) -> int:
        def count(node: Node) -> int:
            return 1 if node.is_leaf else count(node.left) + count(node.right)  # type: ignore[arg-type]

        return count(self.root) if self.root else 0

    @property
    def depth(self) -> int:
        def d(node: Node) -> int:
            return node.depth if node.is_leaf else max(d(node.left), d(node.right))  # type: ignore[arg-type]

        return d(self.root) if self.root else 0

    def feature_importances(self) -> np.ndarray:
        """Importance = total penurunan impuritas berbobot per fitur (Gini importance)."""
        imp = np.zeros(self.n_features_)

        def walk(node: Node) -> None:
            if node.is_leaf:
                return
            n = node.value.sum()
            nl = node.left.value.sum()   # type: ignore[union-attr]
            nr = node.right.value.sum()  # type: ignore[union-attr]
            dec = (
                n * node.impurity
                - nl * node.left.impurity   # type: ignore[union-attr]
                - nr * node.right.impurity  # type: ignore[union-attr]
            )
            imp[node.feature] += dec  # type: ignore[index]
            walk(node.left)   # type: ignore[arg-type]
            walk(node.right)  # type: ignore[arg-type]

        if self.root:
            walk(self.root)
        s = imp.sum()
        return imp / s if s > 0 else imp

    def export_text(self, max_depth: Optional[int] = None) -> str:
        """Cetak struktur pohon sebagai teks bercabang."""
        names = self.feature_names_ or [f"x[{i}]" for i in range(self.n_features_)]
        lines: List[str] = []

        def walk(node: Node, prefix: str, is_last: bool, label: str) -> None:
            if max_depth is not None and node.depth > max_depth:
                return
            branch = "`-- " if is_last else "|-- "
            if node.is_leaf:
                lines.append(
                    f"{prefix}{branch}{label}LEAF  n={node.n_samples} "
                    f"p(1)={node.proba:.3f} -> kelas {node.prediction}"
                )
                return
            lines.append(
                f"{prefix}{branch}{label}{names[node.feature]} <= {node.threshold:.4g}"
                f"   (n={node.n_samples}, imp={node.impurity:.4f})"
            )
            child_prefix = prefix + ("    " if is_last else "|   ")
            walk(node.left, child_prefix, False, "[ya]  ")    # type: ignore[arg-type]
            walk(node.right, child_prefix, True, "[tdk] ")    # type: ignore[arg-type]

        if self.root is not None:
            walk(self.root, "", True, "")
        return "\n".join(lines)

    def get_params(self) -> Dict[str, object]:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "class_weight": self.class_weight,
            "ccp_alpha": self.ccp_alpha,
        }
