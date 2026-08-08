"""
Support Vector Machine dari nol (numpy saja).

Dua implementasi disediakan.

1. ``LinearSVM`` — SVM soft-margin primal
   ---------------------------------------
   Label diubah ke ``t = 2y - 1 in {-1, +1}``, lalu diminimasi::

       J(w, b) = (lambda/2) ||w||^2 + 1/n * sum_i c_i * max(0, 1 - t_i (w^T x_i + b))

   Sub-gradien hinge loss::

       dJ/dw = lambda*w - 1/n * sum_{i in M} c_i t_i x_i     (M = margin violator)
       dJ/db =           - 1/n * sum_{i in M} c_i t_i

   Optimizer:

   * ``"pegasos"`` — **bonus**, *Primal Estimated sub-GrAdient SOlver for SVM*
     (Shalev-Shwartz, Singer, Srebro & Cotter, *Mathematical Programming* 2011).
     Pegasos memakai laju belajar yang meluruh sendiri ``eta_t = 1/(lambda*t)``
     sehingga tidak ada hyperparameter laju belajar yang perlu disetel, lalu
     memproyeksikan ``w`` ke bola berjari-jari ``1/sqrt(lambda)`` — tempat solusi
     optimum dijamin berada. Jumlah iterasi yang dibutuhkan untuk mencapai galat
     ``eps`` adalah ``O(1/(lambda*eps))``, **tidak bergantung pada jumlah data**,
     jadi cocok untuk 28.800 sampel.
   * ``"subgradient"`` — sub-gradient descent mini-batch biasa sebagai pembanding.

2. ``KernelSVM`` — SVM dual dengan SMO
   ------------------------------------
   Implementasi *Simplified Sequential Minimal Optimization* (Platt, 1998):
   dua pengali Lagrange ``alpha_i, alpha_j`` dioptimasi analitik pada tiap
   langkah sambil menjaga kendala ``sum alpha_i t_i = 0`` dan ``0 <= alpha <= C``.
   Kompleksitasnya O(n^2) untuk matriks Gram, sehingga di sini hanya dipakai pada
   **subsampel** untuk memverifikasi bahwa implementasi dual bekerja; SVM
   non-linier pada data penuh dijalankan lewat Random Fourier Features
   (lihat ``preprocessing.RandomFourierFeatures``) + ``LinearSVM``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# SVM primal
# --------------------------------------------------------------------------- #


class LinearSVM:
    """SVM linier soft-margin, dioptimasi di ruang primal.

    Parameter
    ---------
    lambda_
        Koefisien regularisasi (setara ``1/(C*n)`` pada notasi C).
    n_iter
        Jumlah epoch.
    optimizer
        ``"pegasos"`` atau ``"subgradient"``.
    batch_size
        Ukuran mini-batch.
    lr
        Laju belajar untuk ``"subgradient"`` (diabaikan oleh Pegasos).
    class_weight
        ``None`` atau ``"balanced"``.
    """

    def __init__(
        self,
        lambda_: float = 1e-4,
        n_iter: int = 30,
        optimizer: str = "pegasos",
        batch_size: int = 256,
        lr: float = 0.01,
        class_weight: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

    # -- galat ------------------------------------------------------------- #

    def _objective(self, X: np.ndarray, t: np.ndarray, cw: np.ndarray) -> float:
        margins = np.maximum(0.0, 1.0 - t * (X @ self.w + self.b))
        return float(
            0.5 * self.lambda_ * np.dot(self.w, self.w) + np.mean(cw * margins)
        )

    # -- pelatihan --------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        t = np.where(y == 1, 1.0, -1.0)  # label {-1, +1}
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        if self.class_weight == "balanced":
            n1 = float(np.sum(y == 1))
            n0 = float(np.sum(y == 0))
            cw = np.where(y == 1, n / (2 * n1), n / (2 * n0))
        else:
            cw = np.ones(n)

        radius = 1.0 / np.sqrt(self.lambda_)
        step = 0

        for epoch in range(self.n_iter):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                bi = idx[s : s + self.batch_size]
                Xb, tb, cb = X[bi], t[bi], cw[bi]
                step += 1

                margin = tb * (Xb @ self.w + self.b)
                viol = margin < 1.0  # hanya pelanggar margin yang berkontribusi
                k = len(bi)

                grad_w = self.lambda_ * self.w - (Xb[viol].T @ (cb[viol] * tb[viol])) / k
                grad_b = -float(np.sum(cb[viol] * tb[viol]) / k)

                if self.optimizer == "pegasos":
                    eta = 1.0 / (self.lambda_ * step)   # laju belajar meluruh sendiri
                else:
                    eta = self.lr

                self.w -= eta * grad_w
                self.b -= eta * grad_b

                if self.optimizer == "pegasos":
                    # proyeksi ke bola ||w|| <= 1/sqrt(lambda)
                    norm = np.linalg.norm(self.w)
                    if norm > radius:
                        self.w *= radius / norm

            loss = self._objective(X, t, cw)
            self.loss_history.append(loss)
            if self.verbose and (epoch % 5 == 0 or epoch == self.n_iter - 1):
                print(f"   epoch {epoch:>3}  hinge objective={loss:.6f}")

        return self

    # -- inferensi --------------------------------------------------------- #

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return (self.decision_function(X) >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Peluang semu lewat kalibrasi logistik sederhana atas skor margin.

        SVM tidak menghasilkan peluang; transformasi ini hanya dipakai agar
        ROC-AUC dapat dihitung dengan fungsi yang sama seperti model lain
        (monoton, jadi tidak mengubah nilai AUC).
        """
        z = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))

    @property
    def n_support_vectors(self) -> int:
        return -1  # tidak terdefinisi pada formulasi primal

    def get_params(self) -> Dict[str, object]:
        return {
            "lambda_": self.lambda_,
            "n_iter": self.n_iter,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
        }


# --------------------------------------------------------------------------- #
# SVM dual (SMO)
# --------------------------------------------------------------------------- #


class KernelSVM:
    """SVM dual dengan Simplified SMO (Platt, 1998).

    Hanya cocok untuk data berukuran kecil-menengah karena matriks Gram
    berukuran ``n x n`` harus dibentuk.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: float = 0.05,
        degree: int = 3,
        tol: float = 1e-3,
        max_passes: int = 10,
        max_iter: int = 2000,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self.alpha: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.X_: Optional[np.ndarray] = None
        self.t_: Optional[np.ndarray] = None
        self.sv_mask_: Optional[np.ndarray] = None

    # -- kernel ------------------------------------------------------------ #

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "poly":
            return (1.0 + A @ B.T) ** self.degree
        # rbf
        sq = (
            np.sum(A * A, axis=1)[:, None]
            - 2.0 * (A @ B.T)
            + np.sum(B * B, axis=1)[None, :]
        )
        return np.exp(-self.gamma * np.maximum(sq, 0.0))

    # -- pelatihan --------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelSVM":
        X = np.asarray(X, dtype=float)
        t = np.where(np.asarray(y).astype(int) == 1, 1.0, -1.0)
        n = len(t)
        rng = np.random.default_rng(self.random_state)

        K = self._kernel(X, X)
        alpha = np.zeros(n)
        b = 0.0
        passes = 0
        it = 0

        while passes < self.max_passes and it < self.max_iter:
            it += 1
            changed = 0
            f = (alpha * t) @ K + b
            E = f - t

            for i in range(n):
                if (t[i] * E[i] < -self.tol and alpha[i] < self.C) or (
                    t[i] * E[i] > self.tol and alpha[i] > 0
                ):
                    j = rng.integers(n)
                    while j == i:
                        j = rng.integers(n)
                    j = int(j)

                    ai_old, aj_old = alpha[i], alpha[j]
                    if t[i] != t[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)
                    if L >= H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    Ei = float((alpha * t) @ K[:, i] + b - t[i])
                    Ej = float((alpha * t) @ K[:, j] + b - t[j])

                    aj = aj_old - t[j] * (Ei - Ej) / eta
                    aj = float(np.clip(aj, L, H))
                    if abs(aj - aj_old) < 1e-5:
                        continue
                    ai = ai_old + t[i] * t[j] * (aj_old - aj)

                    b1 = (
                        b - Ei
                        - t[i] * (ai - ai_old) * K[i, i]
                        - t[j] * (aj - aj_old) * K[i, j]
                    )
                    b2 = (
                        b - Ej
                        - t[i] * (ai - ai_old) * K[i, j]
                        - t[j] * (aj - aj_old) * K[j, j]
                    )
                    if 0 < ai < self.C:
                        b = b1
                    elif 0 < aj < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    alpha[i], alpha[j] = ai, aj
                    changed += 1

            passes = passes + 1 if changed == 0 else 0
            if self.verbose:
                print(f"   iter {it:>4}  pasangan berubah={changed}  #SV={int(np.sum(alpha > 1e-8))}")

        self.alpha = alpha
        self.b = float(b)
        self.X_ = X
        self.t_ = t
        self.sv_mask_ = alpha > 1e-8
        return self

    # -- inferensi --------------------------------------------------------- #

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        assert self.alpha is not None and self.X_ is not None and self.t_ is not None
        sv = self.sv_mask_
        K = self._kernel(np.asarray(X, dtype=float), self.X_[sv])
        return K @ (self.alpha[sv] * self.t_[sv]) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))

    @property
    def n_support_vectors(self) -> int:
        return int(np.sum(self.sv_mask_)) if self.sv_mask_ is not None else 0

    def get_params(self) -> Dict[str, object]:
        return {"C": self.C, "kernel": self.kernel, "gamma": self.gamma}
