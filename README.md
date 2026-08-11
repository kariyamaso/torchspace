# TorchSpace PoC (v0.1.0.dev0)

Proof-of-concept for **TorchSpace** — a spatial debugger for PyTorch that
combines model architecture, tensor geometry, forward activations, and
backward gradients in one interactive Three.js scene — built as a
**non-invasive extension of [torchview](https://github.com/mert-kurttutan/torchview)**.

This PoC accompanies *TorchSpace Detailed Engineering Design v0.2* and
validates its core architectural claims:

1. torchview's tracer can be extended **without forking or copying code**
   (two scoped composition points: a wrapped module-forward and a composed
   `RecorderTensor.__torch_function__`).
2. A neutral, stable-ID **IR** can be built from the enriched trace and is
   sufficient to drive a renderer with no Python/graphviz dependency.
3. Runtime activation/gradient statistics captured by ordinary hooks on
   *plain* tensors align with the structural trace via `(fqn, call_index)`.
4. The semantic-3D encoding (X/Y architecture · +Z activation · −Z gradient)
   makes vanishing gradients, activation explosions, dead regions, and
   forward/backward asymmetry visible in a single scene.

## Quick start

```bash
pip install dist/torchspace-0.1.0a1-py3-none-any.whl   # or, once published: pip install torchspace
python build_demo.py                        # traces both demo models
# -> out/torchspace_viewer.html   (self-contained; open in any browser)
# -> out/<Model>.torchspace.json  (IR documents)
python tests/test_smoke.py
```

## Notebooks (Colab / Jupyter / VS Code)

```python
run = torchspace.trace(model, input_data=x)
loss = criterion(model(x), y); run.capture_backward(loss)
run.show(height=620)      # full interactive 3D scene, inline in the cell
```

`show()` renders a sandboxed `srcdoc` iframe: three.js is inlined and the IR is
embedded, so it works in Colab's network-restricted output frames and offline.

**Tutorial**: [`TorchSpace_Tutorial.ipynb`](TorchSpace_Tutorial.ipynb) walks
through the full API — quick start, diagnosing a pathological network, the IR,
meta-device inspection, torchview-parity graphviz export, attributes, training
loop instrumentation, and export/sharing. Open it directly in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-GITHUB-USERNAME/torchspace/blob/main/TorchSpace_Tutorial.ipynb)

## API sketch (spec §5)

```python
import torchspace

run = torchspace.trace(model, input_data=x)   # structure + instrument
out = model(x)                                # captured forward
run.capture_backward(criterion(out, y))       # captured backward
run.export_html("model.torchspace.html")
run.detach()

torchspace.view(model, input_size=(1, 3, 224, 224), device="meta")
# structure-only mode, no activation capture (works on meta device)
```

### torchview parity

The trace captures at full detail (torchview's information is a strict
subset of the IR), module/function attributes are collected by default
(`collect_attributes=False` to disable), and torchview's native graphviz
output stays available at any granularity:

```python
run.export_dot("model.gv", depth=3, graph_dir="LR")   # DOT source
run.draw_graph(depth=2, roll=True, save_graph=True,
               filename="model", format="png")        # torchview pipeline
```

`draw_graph`/`export_dot` accept every `torchview.draw_graph` option
(`depth`, `hide_inner_tensors`, `hide_module_functions`, `roll`,
`graph_dir`, `strict`, ...) and reuse the exact inputs of the original
trace; runtime capture hooks are suspended for the auxiliary pass.

### What ends up in an export (privacy)

Exports contain **statistics only** — never raw tensors. Be aware of two
edges before sharing an export: `collect_attributes=True` (default)
embeds a `repr`-style summary of each module's public attributes, which
for custom modules can include checkpoint paths or config values — pass
`collect_attributes=False` for sensitive models; and for a 1-element
tensor the stats (min=max=mean) necessarily reveal its exact value.

## Layout

```
torchspace/
  _tv_compat.py    # single audited import surface over torchview
  tracer.py        # non-invasive structural tracer (extension points A & B)
  runtime.py       # forward/backward hook capture on plain tensors
  stats.py         # compact tensor statistics (never raw tensors)
  ir.py            # IR builder: stable IDs, empty-pass contraction, edges
  diagnostics.py   # rule-based anomaly detection (spec §12)
  api.py           # public view()/trace()/SpaceRun (+ draw_graph/export_dot)
  export.py        # embeds IR into the self-contained viewer
  assets/viewer.html   # packaged viewer (generated — do not edit)
viewer/
  src.html               # viewer source (edit this)
  three_r128_inline.html # vendored three.js r128 (script-wrapped)
scripts/build_viewer.py  # src.html + three -> torchspace/assets/viewer.html
demos.py           # PathologicalMLP + TinyResNet (spec §15)
build_demo.py      # end-to-end driver
tests/test_smoke.py
shot.py            # Playwright screenshot harness
```

After editing `viewer/src.html`, run `python scripts/build_viewer.py` to
regenerate the packaged asset.

## Viewer controls

Drag = rotate · wheel = zoom · right-drag = pan · hover = exact stats ·
click = inspect · double-click = expand/collapse containers · **2D Ortho** =
torchview-style top-down architecture view · timeline = forward/backward
replay with execution-flow pulses (green fwd / purple bwd).
Toggles: Tensor geometry, Activation +Z, Gradient −Z, Show ops,
Roll repeated, Labels, ±Z scale. The log-scale ruler beside the model gives
the quantitative 1e^x scale for both overlays.

## Known PoC limitations (addressed in the design document)

- Layout is a simple deterministic layered algorithm; production uses ELK
  compound layout (containers never overlap siblings). The 3D viewer has
  no `graph_dir` equivalent (use `export_dot(graph_dir=...)` for that).
- Tensor dtype/bytes are captured for op outputs and inputs, not for
  tensors created by `torch.*` creation ops inside `forward`.
- `requires_grad` in the IR reflects the (no-grad) structural pass.
- Diagnostics use median-relative thresholds; monotonic profiles
  (e.g. a fully vanishing chain) need trend-aware rules (v0.2).
- Single step / single device; no live server (static HTML export only).

License: MIT (see `LICENSE`). torchview is MIT-licensed and imported as a
dependency — none of its code is copied. three.js r128 (MIT, © Three.js
Authors) is bundled inside the packaged viewer with its license header
retained.
