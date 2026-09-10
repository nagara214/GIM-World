"""
Information-Guided Pruning (paper Sec. 3.4).

Before memory encoding, the history H is reduced to a retained subset S with
|S| <= K that is maximally informative about the frames it leaves out:

    S* = argmax_S  I(S; H \\ S)   s.t. |S| <= K            (Eq. 15)

A Gaussian process over per-frame observations with the pose-time kernel

    k(c_i, c_j) = exp(-||p_i - p_j||^2 / 2 sigma_p^2
                      - angle(f_i, f_j)^2 / 2 sigma_r^2
                      - (t_i - t_j)^2   / 2 sigma_t^2)      (Eq. 16)

turns the objective into posterior variances (Eq. 17); it is submodular, so
the greedy rule with marginal gain

    0.5 * log( sigma^2(h | S) / sigma^2(h | H \\ S \\ {h}) )   (Eq. 18)

is near-optimal (Krause, Singh & Guestrin, JMLR 2008). Selected indices are
always returned in ascending temporal order.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
from scipy.linalg import solve_triangular

from gim.utils.camera import POSITION_SCALE


def _camera_features(camera_poses: torch.Tensor, indices: Sequence[int]):
    rt = camera_poses.detach().cpu().numpy().astype(np.float64)
    pos = rt[indices, :3] / POSITION_SCALE
    R = rt[indices, 3:].reshape(-1, 3, 3)
    fwd = R[:, :, 2].copy()
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-10)
    return pos, fwd


def _median_positive(x: np.ndarray) -> float:
    pos = x[x > 1e-10]
    return float(np.median(pos)) if pos.size else 1.0


def pose_time_kernel(
    camera_poses: torch.Tensor,
    indices: Sequence[int],
    sigma_p: Optional[float] = None,
    sigma_r: Optional[float] = None,
    sigma_t: Optional[float] = None,
    jitter: float = 1e-4,
) -> np.ndarray:
    """N x N PSD kernel of Eq. 16 over the given history frames.

    sigma_p / sigma_r default to median-heuristic bandwidths; sigma_t <= 0 or
    None disables the temporal factor. A small jitter keeps K invertible.
    """
    pos, fwd = _camera_features(camera_poses, indices)
    n = pos.shape[0]

    d_pos = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    d_ang = np.arccos(np.clip(fwd @ fwd.T, -1.0, 1.0))

    iu = np.triu_indices(n, k=1)
    if sigma_p is None:
        sigma_p = _median_positive(d_pos[iu])
    if sigma_r is None:
        sigma_r = _median_positive(d_ang[iu])

    K = np.exp(-0.5 * (d_pos / sigma_p) ** 2) * np.exp(-0.5 * (d_ang / sigma_r) ** 2)
    if sigma_t is not None and sigma_t > 0:
        t = np.asarray(indices, dtype=np.float64)
        K = K * np.exp(-0.5 * (np.abs(t[:, None] - t[None, :]) / sigma_t) ** 2)

    K = 0.5 * (K + K.T)
    K[np.diag_indices_from(K)] += jitter
    return K


def uniform_pruning(pool_indices: Sequence[int], k: int) -> List[int]:
    """Uniform temporal downsampling (baseline)."""
    n = len(pool_indices)
    if k >= n:
        return list(pool_indices)
    idx = np.linspace(0, n - 1, k, dtype=int)
    return [pool_indices[i] for i in idx]


def _greedy_mi(K: np.ndarray, k: int, force_first=True, force_last=True) -> List[int]:
    """
    Greedy maximization of I(S; V \\ S) under kernel K.

    Maintains a Cholesky factor of K[S, S] (rank-1 growth) and the inverse of
    K[V\\S, V\\S] (Schur downdate), giving vectorized marginal gains per step.

    Anchors: the first pool frame keeps the relative-pose origin fixed; the
    last pool frame guarantees the most recent observation stays in memory.
    """
    n = K.shape[0]
    if k >= n:
        return list(range(n))

    diag_K = np.diag(K).copy()
    bar = np.arange(n)
    M = np.linalg.inv(K)

    selected: List[int] = []
    L = np.zeros((0, 0))
    KSj = np.zeros((0, 0))

    anchors = {}
    if force_first and k >= 1:
        anchors[0] = 0
    if force_last and n >= 2:
        step = 1 if force_first else 0
        if step < k:
            anchors[step] = n - 1

    for step in range(k):
        m = len(bar)
        if m == 0:
            break

        anchor = anchors.get(step)
        if anchor is not None:
            hits = np.where(bar == anchor)[0]
            if hits.size:
                local, chosen = int(hits[0]), int(anchor)
            else:
                anchor = None

        if anchor is None:
            if not selected:
                var_s = diag_K[bar]
            else:
                B = solve_triangular(L, KSj, lower=True)
                var_s = np.maximum(diag_K[bar] - (B * B).sum(axis=0), 1e-12)
            var_r = 1.0 / np.maximum(np.diag(M), 1e-12)
            local = int(np.argmax(np.log(var_s) - np.log(var_r)))
            chosen = int(bar[local])

        k_Sy = K[selected, chosen] if selected else np.zeros(0)
        k_yy = float(diag_K[chosen])
        if L.shape[0] == 0:
            L = np.array([[np.sqrt(max(k_yy, 1e-12))]])
        else:
            u = solve_triangular(L, k_Sy, lower=True)
            d = np.sqrt(max(k_yy - float(u @ u), 1e-12))
            L_new = np.zeros((L.shape[0] + 1, L.shape[0] + 1))
            L_new[:-1, :-1] = L
            L_new[-1, :-1] = u
            L_new[-1, -1] = d
            L = L_new

        selected.append(chosen)

        keep = np.arange(m) != local
        col = M[:, local]
        M = M[np.ix_(keep, keep)] - np.outer(col[keep], col[keep]) / M[local, local]
        bar = bar[keep]
        if selected:
            KSj = K[np.ix_(selected, bar)]

    return selected


def information_guided_pruning(
    pool_indices: Sequence[int],
    k: int,
    camera_poses: torch.Tensor,
    max_kernel_size: int = 1500,
    sigma_p: Optional[float] = None,
    sigma_r: Optional[float] = None,
    sigma_t: Optional[float] = None,
    time_sigma_scale: float = 1.0,
    jitter: float = 1e-4,
) -> List[int]:
    """
    Keep k of the pool frames under the MI criterion (Eq. 15-18).

    Pools larger than `max_kernel_size` are first thinned uniformly to bound
    the O(N^3) kernel inverse. The temporal bandwidth defaults to
    time_sigma_scale * (pool span) / k, i.e. the ideal uniform gap.
    """
    n = len(pool_indices)
    if k >= n:
        return list(pool_indices)

    pool = uniform_pruning(pool_indices, max_kernel_size) if n > max_kernel_size \
        else list(pool_indices)

    if sigma_t is None and time_sigma_scale and time_sigma_scale > 0:
        span = float(max(pool) - min(pool))
        if span > 0 and k > 0:
            sigma_t = time_sigma_scale * span / float(k)

    K = pose_time_kernel(camera_poses, pool, sigma_p=sigma_p, sigma_r=sigma_r,
                         sigma_t=sigma_t, jitter=jitter)
    local = _greedy_mi(K, k, force_first=True, force_last=True)
    return sorted(pool[i] for i in local)


PRUNING_METHODS = ("information_guided", "uniform")


def prune_history(
    method: str,
    pool_indices: Sequence[int],
    budget: int,
    camera_poses: Optional[torch.Tensor] = None,
    **kwargs,
) -> List[int]:
    """Select at most `budget` history latents from `pool_indices`."""
    if budget >= len(pool_indices):
        return list(pool_indices)
    if method == "uniform":
        return uniform_pruning(pool_indices, budget)
    if method == "information_guided":
        if camera_poses is None:
            raise ValueError("information_guided pruning requires camera_poses.")
        return information_guided_pruning(pool_indices, budget, camera_poses, **kwargs)
    raise ValueError(f"Unknown pruning method {method!r}; expected one of {PRUNING_METHODS}.")
