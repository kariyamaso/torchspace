"""Rule-based diagnostics (spec §12): transparent, thresholded, explainable.

Every rule reads only the IR (frames + graph) and emits warnings with the
evidence used, so the UI can show *why* a node is flagged. Thresholds are
deliberately simple for v0.1 and centralised in DEFAULTS.
"""
from __future__ import annotations

import math
from typing import Any

DEFAULTS = dict(
    dead_zero_frac=0.90,       # fraction of exact zeros in activation
    dead_rms=1e-7,             # activation RMS below => dead
    explode_act_logratio=2.0,  # log10 distance from median activation RMS
    explode_act_absmax=1e4,
    vanish_grad_rms=1e-9,      # absolute floor
    vanish_grad_logratio=4.0,  # log10 below median gradient RMS
    explode_grad_logratio=4.0,
    explode_grad_absmax=1e4,
    explode_grad_rms_floor=0.05,  # absolute significance floor: a gradient
                                  # can only "explode" if it is also large
                                  # in absolute terms (median-relative tests
                                  # misfire on monotonically vanishing chains)
    asym_logratio=6.0,         # |log10 act_rms - log10 grad_rms|
)

EPS = 1e-30


def _log10(x: float | None) -> float | None:
    if x is None or x <= 0:
        return None
    return math.log10(x + EPS)


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run_rules(ir: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    """Appends warnings to ir['warnings'] in place."""
    cfg = {**DEFAULTS, **(cfg or {})}
    frames = ir.get("frames", [])
    if not frames:
        return
    last_step = max(f["step"] for f in frames)
    leaf = {n["id"] for n in ir["nodes"]
            if n["kind"] == "module" and n.get("is_leaf_module")}

    act: dict[str, dict[str, Any]] = {}
    grad: dict[str, dict[str, Any]] = {}
    for f in frames:
        if f["step"] != last_step or f["node"] not in leaf:
            continue
        if f["kind"] == "activation":
            act[f["node"]] = f["stats"]          # last output wins
        elif f["kind"] == "gradient":
            g = grad.get(f["node"])
            if g is None or (f["stats"].get("rms") or 0) > (g.get("rms") or 0):
                grad[f["node"]] = f["stats"]

    med_act = _median([v for s in act.values()
                       if (v := _log10(s.get("rms"))) is not None])
    med_grad = _median([v for s in grad.values()
                        if (v := _log10(s.get("rms"))) is not None])
    warns = ir.setdefault("warnings", [])

    def warn(rule: str, sev: str, node: str, msg: str, **ev: Any) -> None:
        warns.append(dict(rule=rule, severity=sev, node=node,
                          message=msg, evidence=ev))

    for nid, s in act.items():
        rms, zf = s.get("rms"), s.get("zero_frac")
        if (s.get("nan") or 0) + (s.get("inf") or 0) > 0:
            warn("nonfinite_activation", "error", nid,
                 f"activation contains {s.get('nan')} NaN / {s.get('inf')} Inf",
                 nan=s.get("nan"), inf=s.get("inf"))
        if zf is not None and zf >= cfg["dead_zero_frac"]:
            warn("dead_region", "warning", nid,
                 f"{zf:.0%} of activations are exactly zero", zero_frac=zf)
        elif rms is not None and rms <= cfg["dead_rms"]:
            warn("dead_region", "warning", nid,
                 f"activation RMS {rms:.2e} is ~zero", rms=rms)
        lr = _log10(rms)
        if (lr is not None and med_act is not None
                and lr - med_act >= cfg["explode_act_logratio"]):
            warn("activation_explosion", "warning", nid,
                 f"activation RMS {rms:.3g} is >=10^{cfg['explode_act_logratio']:.0f}"
                 f" above layer median", rms=rms, median_log10=med_act)
        elif (s.get("absmax") or 0) >= cfg["explode_act_absmax"]:
            warn("activation_explosion", "warning", nid,
                 f"activation |max| {s['absmax']:.3g} exceeds "
                 f"{cfg['explode_act_absmax']:.0e}",
                 absmax=s.get("absmax"))

    for nid, s in grad.items():
        rms = s.get("rms")
        if (s.get("nan") or 0) + (s.get("inf") or 0) > 0:
            warn("nonfinite_gradient", "error", nid,
                 f"gradient contains {s.get('nan')} NaN / {s.get('inf')} Inf",
                 nan=s.get("nan"), inf=s.get("inf"))
        lr = _log10(rms)
        if rms is not None and (
                rms <= cfg["vanish_grad_rms"] or
                (lr is not None and med_grad is not None
                 and med_grad - lr >= cfg["vanish_grad_logratio"])):
            warn("vanishing_gradient", "warning", nid,
                 f"gradient RMS {rms:.2e} vanishes vs median 10^{med_grad:.1f}"
                 if med_grad is not None else
                 f"gradient RMS {rms:.2e} is ~zero",
                 rms=rms, median_log10=med_grad)
        if (lr is not None and med_grad is not None
                and lr - med_grad >= cfg["explode_grad_logratio"]
                and (rms or 0) >= cfg["explode_grad_rms_floor"]):
            warn("exploding_gradient", "warning", nid,
                 f"gradient RMS {rms:.3g} explodes vs layer median",
                 rms=rms, median_log10=med_grad)
        elif (s.get("absmax") or 0) >= cfg["explode_grad_absmax"]:
            warn("exploding_gradient", "warning", nid,
                 f"gradient |max| {s['absmax']:.3g} exceeds "
                 f"{cfg['explode_grad_absmax']:.0e}",
                 absmax=s.get("absmax"))

    for nid in set(act) & set(grad):
        la, lg = _log10(act[nid].get("rms")), _log10(grad[nid].get("rms"))
        if la is not None and lg is not None and \
                abs(la - lg) >= cfg["asym_logratio"]:
            warn("forward_backward_asymmetry", "info", nid,
                 f"|log10 act RMS − log10 grad RMS| = {abs(la - lg):.1f}",
                 act_log10=la, grad_log10=lg)
