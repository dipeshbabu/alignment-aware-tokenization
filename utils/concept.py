from __future__ import annotations

import numpy as np
import torch


def normalize_probe_basis(probe: np.ndarray) -> np.ndarray:
    """Return a row-normalized concept basis with shape [K, D]."""
    arr = np.asarray(probe, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D probe array, got shape {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    keep = norms.reshape(-1) > 1e-12
    if not np.any(keep):
        raise ValueError("Probe basis has no nonzero rows")
    arr = arr[keep] / (norms[keep] + 1e-12)
    return arr.astype(np.float32, copy=False)


def concept_scores_from_pooled(pooled: torch.Tensor, probe: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Score pooled hidden states against a vector or subspace probe.

    Rank-1 probes preserve the signed projection used by the original AAT
    implementation. Rank-k probes return the L2 projection energy into the
    hazard subspace, which captures multi-faceted hazard evidence without
    privileging a single direction.
    """
    device = pooled.device
    if isinstance(probe, np.ndarray):
        basis = torch.as_tensor(normalize_probe_basis(probe), dtype=torch.float32, device=device)
    else:
        basis = probe.to(device=device, dtype=torch.float32)
        if basis.ndim == 1:
            basis = basis.view(1, -1)
        basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-9)

    pooled = pooled.to(torch.float32)
    proj = pooled @ basis.t()
    if basis.shape[0] == 1:
        return proj[:, 0]
    return torch.linalg.vector_norm(proj, ord=2, dim=-1)


def append_residual_axes(
    basis: list[np.ndarray],
    X_hazard: np.ndarray,
    X_neutral: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Append orthogonal hazard-residual axes until `rank` is reached."""
    if not basis:
        raise ValueError("basis must contain at least the discriminative axis")
    rows = [np.asarray(v, dtype=np.float32).reshape(-1) for v in basis]
    rows = [v / (np.linalg.norm(v) + 1e-12) for v in rows]

    residual = X_hazard.astype(np.float32) - X_neutral.astype(np.float32).mean(axis=0, keepdims=True)
    for v in rows:
        residual = residual - (residual @ v.reshape(-1, 1)) * v.reshape(1, -1)
    residual = residual - residual.mean(axis=0, keepdims=True)

    if rank > len(rows) and residual.shape[0] > 1:
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        for cand in vt:
            v = cand.astype(np.float32)
            for prev in rows:
                v = v - float(np.dot(v, prev)) * prev
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                rows.append(v / n)
            if len(rows) >= rank:
                break

    return normalize_probe_basis(np.stack(rows, axis=0))
