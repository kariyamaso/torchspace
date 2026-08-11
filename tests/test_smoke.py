"""PoC smoke tests: structural invariants of the produced IR."""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchspace
from torchspace.demos import PathologicalMLP, TinyResNet


def _ir_for(model, x, name):
    run = torchspace.trace(model, input_data=x, model_name=name)
    out = model(x)
    run.capture_backward(out.sum())
    run.detach()
    return run.ir


def check(ir):
    ids = [n["id"] for n in ir["nodes"]]
    assert len(ids) == len(set(ids)), "node ids must be unique"
    idset = set(ids)
    for e in ir["edges"]:
        assert e["source"] in idset and e["target"] in idset
        assert e["tensor"] in {t["id"] for t in ir["tensors"]}
    for n in ir["nodes"]:
        assert n["parent"] is None or n["parent"] in idset
    mods = [n for n in ir["nodes"] if n["kind"] == "module"]
    assert all(m.get("seq") is not None for m in mods), "exec seq stamped"
    # frame alignment: every frame refers to an existing node
    for f in ir["frames"]:
        assert f["node"] in idset
    align = [w for w in ir["warnings"] if w["rule"] == "alignment"]
    assert not align, f"alignment warnings: {align}"
    # determinism of ids: modules keyed by fqn
    root = [n for n in mods if n["parent"] is None]
    assert len(root) == 1


def test_mlp():
    torch.manual_seed(0)                      # seed BEFORE model init
    ir = _ir_for(PathologicalMLP(), torch.randn(8, 64), "MLP")
    check(ir)
    rules = {w["rule"] for w in ir["warnings"]}
    assert "vanishing_gradient" in rules and "dead_region" in rules


def test_resnet():
    torch.manual_seed(0)
    ir = _ir_for(TinyResNet().eval(), torch.randn(1, 3, 32, 32), "TinyResNet")
    check(ir)
    adds = [n for n in ir["nodes"] if n["kind"] == "op" and n["op"] == "add"]
    assert len(adds) == 4, "one residual add per BasicBlock"
    # each add has >= 2 incoming edges (identity + conv path)
    for a in adds:
        inc = [e for e in ir["edges"] if e["target"] == a["id"]]
        assert len(inc) >= 2, f"residual join missing an input: {a['id']}"


def test_structure_only_meta():
    ir = torchspace.view(TinyResNet(), input_size=(1, 3, 32, 32),
                         device="meta", model_name="meta").ir
    assert ir["frames"] == []
    assert any(n["kind"] == "module" for n in ir["nodes"])


def test_collect_attributes():
    torch.manual_seed(0)
    ir = _ir_for(TinyResNet().eval(), torch.randn(1, 3, 32, 32), "attrs")
    convs = [n for n in ir["nodes"]
             if n["kind"] == "module" and n["op"] == "Conv2d"]
    assert convs and all(n.get("attrs") for n in convs)
    assert any("kernel_size" in n["attrs"] for n in convs)


def test_export_dot_parity():
    """torchview parity: dot output at torchview's own default granularity."""
    torch.manual_seed(0)
    model = TinyResNet().eval()
    x = torch.randn(1, 3, 32, 32)
    run = torchspace.trace(model, input_data=x, model_name="dot")
    n_frames = len(run.ir["frames"])
    src = run.export_dot(depth=3, graph_dir="LR")
    run.detach()
    assert src.lstrip().startswith(("digraph", "strict digraph"))
    assert "rankdir=LR" in src
    # the auxiliary pass must not have polluted the frame log
    assert len(run.ir["frames"]) == n_frames


if __name__ == "__main__":
    test_mlp(); test_resnet(); test_structure_only_meta()
    test_collect_attributes(); test_export_dot_parity()
    print("all smoke tests passed")
