"""
Logistic Regression dari nol (numpy saja).

Model
-----
``p(y=1 | x) = sigma(w^T x + b)`` dengan ``sigma(z) = 1 / (1 + e^-z)``.

Fungsi galat yang diminimasi adalah *binary cross-entropy* berbobot kelas
ditambah regularisasi L2::

    J(w, b) = -1/n * sum_i c_{y_i} [ y_i log p_i + (1-y_i) log(1-p_i) ]
              + (lambda/2) ||w||^2

Gradiennya berbentuk sangat sederhana karena turunan sigmoid dan cross-entropy
saling meniadakan::

    dJ/dw = 1/n * X^T (c .* (p - y)) + lambda * w
    dJ/db = 1/n * sum(c .* (p - y))

Tiga *optimizer* disediakan:

``batch``
    Gradient descent penuh — baseline yang diajarkan di kelas.
``sgd``
    Mini-batch stochastic gradient descent.
``adam``
    **Bonus** — Adaptive Moment Estimation (Kingma & Ba, ICLR 2015). Adam
    memelihara rata-rata bergerak momen pertama ``m`` (arah gradien) dan momen
    kedua ``v`` (skala kuadrat gradien), lalu melangkah sebesar
    ``alpha * m_hat / (sqrt(v_hat) + eps)``. Karena setiap parameter memperoleh
    laju belajar efektifnya sendiri, Adam jauh lebih tahan terhadap perbedaan
    skala antar fitur dan konvergen dalam epoch yang jauh lebih sedikit
    dibanding batch GD dengan satu laju belajar global.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid yang stabil secara numerik (tanpa overflow untuk |z| besar)."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """Regresi logistik biner.

    Parameter
    ---------
    lr
        Laju belajar.
    n_iter
        Jumlah epoch (satu epoch = satu lintasan penuh atas data latih).
    l2
        Koefisien regularisasi L2 (``lambda``).
    optimizer
        ``"batch"``, ``"sgd"``, atau ``"adam"``.
    batch_size
        Ukuran mini-batch untuk ``sgd``/``adam``.
    class_weight
        ``None`` atau ``"balanced"``.
    tol
        Ambang penghentian dini berdasarkan perubahan galat antar epoch.
    """

    def __init__(
        self,
        lr: float = 0.1,
        n_iter: int = 300,
        l2: float = 1e-4,
        optimizer: str = "adam",
        batch_size: int = 512,
        class_weight: Optional[str] = None,
        tol: float = 1e-7,
        fit_intercept: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.random_state = random_state
        self.verbose = verbose

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []
        self.weight_path: List[np.ndarray] = []   # lintasan parameter (untuk visualisasi)
        self.n_epochs_run: int = 0

    # -- galat ------------------------------------------------------------- #

    def _loss(self, X: np.ndarray, y: np.ndarray, cw: np.ndarray) -> float:
        p = sigmoid(X @ self.w + self.b)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ce = -(cw * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean()
        return float(ce + 0.5 * self.l2 * np.dot(self.w, self.w))

    # -- pelatihan --------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray, track_path: bool = False) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []
        self.weight_path = []

        if self.class_weight == "balanced":
            n1 = float(np.sum(y == 1))
            n0 = float(np.sum(y == 0))
            w_pos, w_neg = n / (2 * n1), n / (2 * n0)
        else:
            w_pos = w_neg = 1.0
        cw = np.where(y == 1, w_pos, w_neg)

        m_w = np.zeros(d)
        v_w = np.zeros(d)
        m_b = v_b = 0.0
        step = 0
        prev = np.inf

        for epoch in range(self.n_iter):
            if self.optimizer == "batch":
                batches = [np.arange(n)]
            else:
                idx = rng.permutation(n)
                batches = [
                    idx[i : i + self.batch_size] for i in range(0, n, self.batch_size)
                ]

            for bi in batches:
                Xb, yb, cb = X[bi], y[bi], cw[bi]
                p = sigmoid(Xb @ self.w + self.b)
                err = cb * (p - yb)
                gw = Xb.T @ err / len(bi) + self.l2 * self.w
                gb = float(err.mean())

                if not self.fit_intercept:
                    gb = 0.0

                if self.optimizer == "adam":
                    step += 1
                    m_w = self.beta1 * m_w + (1 - self.beta1) * gw
                    v_w = self.beta2 * v_w + (1 - self.beta2) * (gw * gw)
                    m_b = self.beta1 * m_b + (1 - self.beta1) * gb
                    v_b = self.beta2 * v_b + (1 - self.beta2) * (gb * gb)
                    mh_w = m_w / (1 - self.beta1**step)
                    vh_w = v_w / (1 - self.beta2**step)
                    mh_b = m_b / (1 - self.beta1**step)
                    vh_b = v_b / (1 - self.beta2**step)
                    self.w -= self.lr * mh_w / (np.sqrt(vh_w) + self.eps)
                    if self.fit_intercept:
                        self.b -= self.lr * mh_b / (np.sqrt(vh_b) + self.eps)
                else:
                    self.w -= self.lr * gw
                    if self.fit_intercept:
                        self.b -= self.lr * gb

            loss = self._loss(X, y, cw)
            self.loss_history.append(loss)
            if track_path:
                self.weight_path.append(np.concatenate([[self.b], self.w.copy()]))
            self.n_epochs_run = epoch + 1
            if self.verbose and (epoch % 25 == 0 or epoch == self.n_iter - 1):
                print(f"   epoch {epoch:>4}  loss={loss:.6f}")
            if abs(prev - loss) < self.tol:
                break
            prev = loss

        return self

    # -- inferensi --------------------------------------------------------- #

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def coefficients(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        names = feature_names or [f"x[{i}]" for i in range(len(self.w))]
        return {"(intercept)": self.b, **dict(zip(names, self.w.tolist()))}

    def get_params(self) -> Dict[str, object]:
        return {
            "lr": self.lr,
            "n_iter": self.n_iter,
            "l2": self.l2,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
        }
