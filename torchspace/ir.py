"""TorchSpace IR builder: converts an enriched torchview ComputationGraph
(+ optional runtime frames) into the neutral, JSON-serializable IR that is
the contract between the Python side and the Three.js renderer (spec §8/§9).

Stable ID scheme (deterministic for a fixed model + input + seed):
  modules   ->  fqn ('' becomes the model class name), '#k' suffix for the
                k-th extra call of the same module (k >= 1)
  functions ->  op.{seq}.{name}       (seq = execution order)
  tensors   ->  tensor.{n}            (n = first-encounter order in the walk)
  inputs    ->  input.{i} , outputs -> output.{i}, created consts -> const.{i}

The IR always stores the *unrolled* truth. Rolling repeated blocks, hiding
inner tensors, depth cut-offs etc. are view-side transforms, driven by
`unit_key` (same nn.Module object => same unit_key) which is what
torchview's `roll` uses semantically.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import torch

from ._tv_compat import (
    FunctionNode,
    ModuleNode,
    TensorNode,
    TORCHVIEW_VERSION,
)
from .tracer import TraceResult
from .runtime import RuntimeSession

SCHEMA_VERSION = "0.1.0"


class _UnionFind(dict):
    def find(self, k):
        while self.get(k, k) != k:
            self[k] = self.get(self[k], self[k])
            k = self[k]
        return k

    def union(self, child, parent):
        self[self.find(child)] = self.find(parent)


def build_ir(trace: TraceResult, runtime: Optional[RuntimeSession] = None,
             model_name: Optional[str] = None) -> dict[str, Any]:
    mg = trace.model_graph
    model_name = model_name or type(trace.model).__name__

    nodes: list[dict[str, Any]] = []
    tensors: dict[str, dict[str, Any]] = {}      # keyed by torchview node_id
    module_ids: dict[int, str] = {}              # id(ModuleNode) -> IR id
    func_ids: dict[int, str] = {}                # id(FunctionNode) -> IR id
    by_call_key: dict[tuple[str, int], str] = {} # (fqn, call_idx) -> IR id
    empty_pass: set[int] = set()                 # id(FunctionNode)
    alias = _UnionFind()                         # tensor node_id aliasing
    x_counter = [0]

    # ---------------- pass 1: walk hierarchy, create nodes -------------- #
    def tensor_record(tn: TensorNode) -> str:
        key = tn.node_id
        if key not in tensors:
            meta = getattr(tn, "ts_meta", None) or \
                   getattr(tn.main_node, "ts_meta", None) or {}
            tensors[key] = dict(
                id=f"tensor.{len(tensors)}",
                shape=list(tn.tensor_shape),
                dtype=meta.get("dtype"),
                bytes=meta.get("nbytes"),
                device=meta.get("device"),
                requires_grad=meta.get("requires_grad"),
                _names={tn.name},
                _depth=tn.depth,
                _is_root=tn.is_root() and not tn.is_aux,
                _is_leaf=tn.is_leaf(),
            )
        else:
            tensors[key]["_names"].add(tn.name)
            tensors[key]["_is_leaf"] = tensors[key]["_is_leaf"] and tn.is_leaf()
        return key

    def module_ir_id(mnode: ModuleNode) -> str:
        meta = getattr(mnode, "ts_meta", None) or {}
        fqn = meta.get("fqn", None)
        base = model_name if fqn == "" else (fqn if fqn is not None
                                             else f"anon.{len(module_ids)}")
        k = meta.get("call_index", 0)
        return base if k == 0 else f"{base}#{k}"

    def walk(items: list, parent_id: Optional[str], depth: int) -> None:
        for item in items:
            if isinstance(item, dict):
                mnode, children = next(iter(item.items()))
                mid = module_ir_id(mnode)
                module_ids[id(mnode)] = mid
                meta = getattr(mnode, "ts_meta", None) or {}
                if "fqn" in meta:
                    by_call_key[(meta["fqn"], meta.get("call_index", 0))] = mid
                nodes.append(dict(
                    id=mid, kind="module", op=mnode.name,
                    name=meta.get("fqn") or model_name,
                    parent=parent_id, depth=depth,
                    unit_key=f"unit.{mnode.compute_unit_id}",
                    call_index=meta.get("call_index", 0),
                    seq=meta.get("seq_in"), seq_end=meta.get("seq_out"),
                    time_ns=meta.get("time_ns"),
                    params=meta.get("params_total"),
                    params_own=meta.get("params_own"),
                    trainable=meta.get("trainable"),
                    is_leaf_module=meta.get("is_leaf_module"),
                    aliases=meta.get("aliases", []),
                    attrs=getattr(mnode, "attributes", None),
                    in_shapes=[list(s) for s in mnode.input_shape],
                    out_shapes=[list(s) for s in mnode.output_shape],
                ))
                walk(children, mid, depth + 1)
            elif isinstance(item, FunctionNode):
                if item.name == "empty-pass":
                    empty_pass.add(id(item))
                    continue
                if id(item) in func_ids:
                    continue
                meta = getattr(item, "ts_meta", None) or {}
                seq = meta.get("seq")
                if seq is None:
                    fid = f"op.x{x_counter[0]}.{item.name}"
                    x_counter[0] += 1
                else:
                    fid = f"op.{seq}.{item.name}"
                func_ids[id(item)] = fid
                nodes.append(dict(
                    id=fid, kind="op", op=item.name, name=item.name,
                    parent=parent_id, depth=depth, seq=seq,
                    unit_key=f"fn.{id(item)}",
                    attrs=getattr(item, "attributes", None),
                    in_shapes=[list(s) for s in item.input_shape],
                    out_shapes=[list(s) for s in item.output_shape],
                ))
            elif isinstance(item, TensorNode):
                tensor_record(item)

    root_entry = mg.node_hierarchy
    main_container, top_items = next(iter(root_entry.items()))
    walk(top_items, None, 0)

    # ---------------- pass 2: edges (with empty-pass contraction) ------- #
    raw_prod: list[tuple[int, str]] = []   # (id(func), tensor_key)
    raw_cons: list[tuple[str, int]] = []
    ep_in: dict[int, str] = {}
    ep_out: dict[int, list[str]] = {}

    for tail, head in mg.edge_list:
        if isinstance(tail, TensorNode) and not isinstance(head, TensorNode):
            tk = tensor_record(tail)
            if id(head) in empty_pass:
                ep_in[id(head)] = tk
            else:
                raw_cons.append((tk, id(head)))
        elif isinstance(head, TensorNode) and not isinstance(tail, TensorNode):
            tk = tensor_record(head)
            if id(tail) in empty_pass:
                ep_out.setdefault(id(tail), []).append(tk)
            else:
                raw_prod.append((id(tail), tk))

    for ep, src in ep_in.items():
        for out_k in ep_out.get(ep, []):
            alias.union(out_k, src)

    producers: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    for f, tk in raw_prod:
        rk = alias.find(tk)
        if f in func_ids:
            producers[rk] = func_ids[f]
    for tk, f in raw_cons:
        rk = alias.find(tk)
        if f in func_ids:
            consumers.setdefault(rk, []).append(func_ids[f])

    # canonical tensor list (aliased duplicates merged, names/leaf-ness kept:
    # an alias may carry the "output-tensor" name or graph-leaf status that
    # the canonical record lacks)
    canon: dict[str, dict[str, Any]] = {}
    for key, rec in tensors.items():
        rk = alias.find(key)
        base = canon.get(rk)
        if base is None:
            base = dict(tensors[rk]) if rk in tensors else dict(rec)
            base["_names"] = set(base["_names"])
            canon[rk] = base
        base["_names"] |= rec["_names"]
        base["_is_leaf"] = base["_is_leaf"] or rec["_is_leaf"]
    io_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    n_in = n_out = n_const = 0

    for rk, rec in canon.items():
        rec = dict(rec)
        tid = rec["id"]
        names = rec.pop("_names")
        depth0 = rec.pop("_depth")
        is_root = rec.pop("_is_root")
        is_leaf = rec.pop("_is_leaf")
        prod = producers.get(rk)
        cons = consumers.get(rk, [])
        rec["producer"] = prod
        rec["consumers"] = cons
        canon[rk] = rec

        if is_root and prod is None:
            if "input-tensor" in names and depth0 == 0:
                nid = f"input.{n_in}"; n_in += 1
                io_nodes.append(dict(id=nid, kind="input", op="Input",
                                     name=nid, parent=None, depth=0,
                                     out_shapes=[rec["shape"]], tensor=tid))
            else:
                nid = f"const.{n_const}"; n_const += 1
                io_nodes.append(dict(id=nid, kind="const", op="TensorCreate",
                                     name=nid, parent=None, depth=depth0,
                                     out_shapes=[rec["shape"]], tensor=tid))
            for c in cons:
                edges.append(dict(source=nid, target=c, tensor=tid))
        else:
            for c in cons:
                if prod is not None:
                    edges.append(dict(source=prod, target=c, tensor=tid))
        if ("output-tensor" in names) and is_leaf and prod is not None:
            nid = f"output.{n_out}"; n_out += 1
            io_nodes.append(dict(id=nid, kind="output", op="Output",
                                 name=nid, parent=None, depth=0,
                                 in_shapes=[rec["shape"]], tensor=tid))
            edges.append(dict(source=prod, target=nid, tensor=tid))

    nodes.extend(io_nodes)

    # de-duplicate edges (parallel identical edges collapse with a count)
    edge_map: dict[tuple[str, str, str], int] = {}
    for e in edges:
        k = (e["source"], e["target"], e["tensor"])
        edge_map[k] = edge_map.get(k, 0) + 1
    edges = [dict(source=s, target=t, tensor=x, count=c)
             for (s, t, x), c in edge_map.items()]

    # ---------------- pass 3: frames from runtime session --------------- #
    frames: list[dict[str, Any]] = []
    align_warnings: list[dict[str, Any]] = []
    if runtime is not None:
        missing: set[tuple[str, int]] = set()
        for fr in runtime.frames:
            key = (fr["fqn"], fr["call_index"])
            nid = by_call_key.get(key)
            if nid is None:
                missing.add(key)
                continue
            frames.append(dict(
                step=fr["step"], seq=fr["seq"], phase=fr["phase"],
                node=nid, kind=fr["kind"], param=fr.get("param"),
                out_index=fr.get("out_index", 0), stats=fr["stats"]))
        for fqn, k in sorted(missing):
            align_warnings.append(dict(
                rule="alignment", severity="warning", node=None,
                message=f"runtime frame for ({fqn!r}, call {k}) has no "
                        f"structural node — dynamic control flow divergence?",
                evidence={}))

    ir = dict(
        schema_version=SCHEMA_VERSION,
        model=dict(
            name=model_name,
            framework="pytorch",
            torch_version=torch.__version__,
            torchview_version=TORCHVIEW_VERSION,
            params_total=sum(p.numel() for p in trace.model.parameters()),
            inputs=trace.input_meta,
        ),
        nodes=nodes,
        tensors=sorted(canon.values(), key=lambda r: int(r["id"].split(".")[1])),
        edges=edges,
        frames=frames,
        warnings=align_warnings,
    )
    return ir


def to_json(ir: dict[str, Any], **kw: Any) -> str:
    return json.dumps(ir, ensure_ascii=False, **kw)
