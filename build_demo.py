"""End-to-end PoC driver: trace demo models, capture forward+backward,
build IR, run diagnostics, export the combined viewer HTML."""
from __future__ import annotations

import json
import os
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torchspace
from torchspace.export import export_html
from torchspace.demos import PathologicalMLP, TinyResNet

torch.manual_seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def run_demo(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             criterion: nn.Module, name: str) -> dict:
    t0 = time.perf_counter()
    run = torchspace.trace(model, input_data=x, model_name=name)
    t_trace = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = model(x)                       # captured forward (frames)
    loss = criterion(out, y)
    run.capture_backward(loss)           # captured backward (frames)
    t_run = time.perf_counter() - t0
    run.detach()

    ir = run.ir
    p = os.path.join(OUT, f"{name}.torchspace.json")
    run.save_ir(p)
    kinds = {}
    for n in ir["nodes"]:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    print(f"== {name}: trace {t_trace*1e3:.0f} ms | fwd+bwd {t_run*1e3:.0f} ms"
          f" | ir {os.path.getsize(p)/1024:.0f} KB")
    print(f"   nodes {kinds} | tensors {len(ir['tensors'])} "
          f"| edges {len(ir['edges'])} | frames {len(ir['frames'])} "
          f"| warnings {len(ir['warnings'])}")
    for w in ir["warnings"][:12]:
        print(f"   [{w['severity']:7s}] {w['rule']:26s} {w['node']}: "
              f"{w['message'][:80]}")
    if len(ir["warnings"]) > 12:
        print(f"   ... and {len(ir['warnings']) - 12} more warnings")
    return ir


def main() -> None:
    irs = []

    mlp = PathologicalMLP()
    x = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))
    irs.append(run_demo(mlp, x, y, nn.CrossEntropyLoss(), "PathologicalMLP"))

    net = TinyResNet().eval()
    xi = torch.randn(1, 3, 32, 32)
    yi = torch.randint(0, 10, (1,))
    irs.append(run_demo(net, xi, yi, nn.CrossEntropyLoss(), "TinyResNet"))

    out = export_html(irs, os.path.join(OUT, "torchspace_viewer.html"),
                      "TorchSpace PoC")
    print(f"viewer -> {out} ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
