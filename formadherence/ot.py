"""Optimal-transport distances between two empirical distributions of frames.

Two backends, selectable via config (Section 6 of the proposal):

  * exact    Wasserstein-1 by solving the transportation LP. Non-differentiable,
             used for evaluation and as the episodic RL reward.
  * sinkhorn Entropic-regularized OT (Cuturi 2013). Differentiable surrogate for
             lower-variance gradients during fine-tuning.

Both operate on point clouds X (n, d) and Y (m, d) with uniform marginals by
default. The ground metric is cosine distance in [0, 2], which is scale
-invariant -- appropriate for normalized foundation-model embeddings.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def cost_matrix(X: np.ndarray, Y: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Ground-metric cost between every pair of frames. Shape (n, m)."""
    C = cdist(X, Y, metric=metric)
    # Guard against tiny negatives from numerical error in cosine.
    return np.clip(C, 0.0, None)


def wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    metric: str = "cosine",
) -> float:
    """Exact Wasserstein-1 distance via the transportation LP.

    Minimizes <P, C> subject to row sums = a, col sums = b, P >= 0.
    Uses scipy HiGHS. Suitable for the frame counts we cap segments at
    (<= max_frames_per_segment), where the LP is small and fast.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        raise ValueError("empty distribution passed to wasserstein_distance")
    a = _uniform(n) if a is None else np.asarray(a, dtype=np.float64)
    b = _uniform(m) if b is None else np.asarray(b, dtype=np.float64)

    C = cost_matrix(X, Y, metric=metric)
    c = C.reshape(-1)

    # Equality constraints: n row-sum + m col-sum constraints on the n*m plan.
    # Row block: each row i sums to a[i].
    A_rows = np.zeros((n, n * m))
    for i in range(n):
        A_rows[i, i * m:(i + 1) * m] = 1.0
    # Col block: each col j sums to b[j].
    A_cols = np.zeros((m, n * m))
    for j in range(m):
        A_cols[j, j::m] = 1.0
    # One equality constraint is redundant (marginals both sum to 1); drop the
    # last column constraint to keep the system full-rank and the LP well posed.
    A_eq = np.vstack([A_rows, A_cols[:-1]])
    b_eq = np.concatenate([a, b[:-1]])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"OT LP failed: {res.message}")
    return float(res.fun)


def sinkhorn_distance(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-6,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    metric: str = "cosine",
    return_plan: bool = False,
):
    """Entropic-regularized OT (Sinkhorn-Knopp) in log domain for stability.

    Returns the transport cost <P, C> under the regularized optimal plan P.
    This is the differentiable surrogate referenced in Section 6; here it is
    implemented in numpy for verification, but the identical recurrence ports
    directly to a torch autograd graph for actual RL fine-tuning.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        raise ValueError("empty distribution passed to sinkhorn_distance")
    a = _uniform(n) if a is None else np.asarray(a, dtype=np.float64)
    b = _uniform(m) if b is None else np.asarray(b, dtype=np.float64)

    C = cost_matrix(X, Y, metric=metric)
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)
    K = -C / eps  # log-kernel

    f = np.zeros(n)  # log-domain scalings
    g = np.zeros(m)
    for _ in range(max_iter):
        f_prev = f
        # f_i = log_a_i - logsumexp_j (K_ij + g_j)
        f = log_a - _logsumexp(K + g[None, :], axis=1)
        g = log_b - _logsumexp(K + f[:, None], axis=0)
        if np.max(np.abs(f - f_prev)) < tol:
            break

    logP = f[:, None] + K + g[None, :]
    P = np.exp(logP)
    cost = float(np.sum(P * C))
    if return_plan:
        return cost, P
    return cost


def _logsumexp(M: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(M, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(M - mx), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)
