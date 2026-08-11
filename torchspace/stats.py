"""Compact tensor statistics — the only thing TorchSpace ships to the browser
by default (never raw tensors). One dict per (node, phase, step).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch


def tensor_stats(t: torch.Tensor) -> Optional[dict[str, Any]]:
    """Single-pass summary statistics of a tensor. Detaches; never keeps a
    reference to the autograd graph. Returns None for empty tensors."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    with torch.no_grad():
        t = t.detach()
        numel = t.numel()
        out: dict[str, Any] = {"numel": int(numel)}
        if t.is_floating_point() or t.is_complex():
            f = t.abs() if t.is_complex() else t
            f = f.float()
            nan = torch.isnan(f).sum()
            inf = torch.isinf(f).sum()
            finite = f[torch.isfinite(f)] if (nan + inf) > 0 else f
            out["nan"] = int(nan)
            out["inf"] = int(inf)
            if finite.numel() == 0:
                out.update(rms=None, mean=None, std=None, min=None, max=None,
                           absmax=None, zero_frac=None, l2=None)
                return out
            sq = finite.square().mean()
            out["rms"] = _f(sq.sqrt())
            out["l2"] = _f(finite.norm(2))
            out["mean"] = _f(finite.mean())
            out["std"] = _f(finite.std(unbiased=False))
            out["min"] = _f(finite.min())
            out["max"] = _f(finite.max())
            out["absmax"] = _f(finite.abs().max())
            out["zero_frac"] = _f((finite == 0).float().mean())
        else:
            out["nan"] = 0
            out["inf"] = 0
            fl = t.float()
            out["rms"] = _f(fl.square().mean().sqrt())
            out["l2"] = _f(fl.norm(2))
            out["mean"] = _f(fl.mean())
            out["std"] = _f(fl.std(unbiased=False)) if numel > 1 else 0.0
            out["min"] = _f(fl.min())
            out["max"] = _f(fl.max())
            out["absmax"] = _f(fl.abs().max())
            out["zero_frac"] = _f((t == 0).float().mean())
        return out


def _f(x: torch.Tensor) -> float | None:
    v = float(x)
    if math.isnan(v):
        return None
    if math.isinf(v):
        return 3.0e38 if v > 0 else -3.0e38
    return v
