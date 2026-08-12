<h1 align="center">TorchSpace</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/kariyamaso/torchspace/main/docs/assets/hero.png" width="100%" alt="TorchSpace viewer — a 24-layer pathological MLP as a 3D scene: green +Z activation bars, one red exploding layer, the diagnostics panel filtered to backbone.4 and the inspector showing exact statistics" />
</p>

<p align="center"><b>Spatial debugging for PyTorch — architecture, activations and gradients in one interactive 3D scene</b></p>

<p align="center">
  <a href="https://colab.research.google.com/github/kariyamaso/torchspace/blob/main/TorchSpace_Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" /></a>
  <a href="https://pypi.org/project/torchspace/"><img src="https://img.shields.io/pypi/v/torchspace?color=2fb3a5" alt="PyPI" /></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+" />
  <a href="https://github.com/kariyamaso/torchspace/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
</p>

<p align="center">
  <a href="https://kariyamaso.github.io/torchspace/"><img src="https://img.shields.io/badge/%E2%96%B6%20Live%20Demo-open%20in%20your%20browser-2fb3a5?style=for-the-badge" alt="▶ Live Demo" /></a>
</p>
<p align="center"><sub>
▲ Live Demo — the full interactive viewer with two demo models, running entirely in your browser (nothing to install, no server)
</sub></p>

> **TorchSpace** renders a PyTorch model as one interactive Three.js scene:
> **X·Y** is the architecture (layers, blocks, skip connections), **+Z** is
> forward activation magnitude and **−Z** is backward gradient magnitude, both
> on a log scale. Vanishing gradients, activation explosions, dead regions and
> forward/backward asymmetry become *visible shapes* instead of numbers you
> have to hunt for. Built as a **non-invasive extension of
> [torchview](https://github.com/mert-kurttutan/torchview)** — no code copied,
> full torchview feature set preserved.

| axis | meaning |
|---|---|
| **X · Y** | architecture — layers, containers, skip connections |
| **+Z** | forward \|activation\| per layer (log scale) |
| **−Z** | backward \|gradient\| per layer (log scale) |

Everything is captured as **summary statistics only** (never raw tensors) and
exported as a single self-contained HTML file: three.js is inlined and the
statistics are embedded, so it works in Colab, Jupyter, VS Code, as an email
attachment, and offline.

---

## Install

```bash
pip install torchspace
```

## Quick start

```python
import torchspace

run = torchspace.trace(model, input_data=x)   # structure + instrumentation
out = model(x)                                # captured forward
run.capture_backward(criterion(out, y))       # captured backward
run.show(height=620)                          # interactive scene, inline in a notebook
run.export_html("model.torchspace.html")      # ...or one shareable file
run.detach()                                  # de-instrument the model

torchspace.view(model, input_size=(1, 3, 224, 224), device="meta")
# structure-only mode, no activation capture (works on meta device)
```

**Tutorial**: [`TorchSpace_Tutorial.ipynb`](https://github.com/kariyamaso/torchspace/blob/main/TorchSpace_Tutorial.ipynb) walks
through the full API — quick start, diagnosing a pathological network, the IR,
meta-device inspection, torchview-parity graphviz export, module attributes,
training-loop instrumentation, and export/sharing. Open it with the Colab
badge at the top of this page.

---

## What you get

- **One scene for the whole story** — architecture, per-layer activation and
  gradient magnitude, execution order and timing, all aligned in space.
- **Rule-based diagnostics with evidence** — vanishing/exploding gradients,
  dead regions, NaN/Inf, forward/backward asymmetry. Click a finding to jump
  to the layer; select a layer to filter the panel to its findings.
- **Timeline replay** — animate the actual forward (green) / backward
  (purple) execution order with flow pulses.
- **Real training loops** — statistics are captured by ordinary hooks on your
  *actual* training steps, on-device, detached immediately. No separate
  profiling pass, no `RecorderTensor` near your hot path.
- **Notebook-native** — `run.show()` renders the scene in a sandboxed
  `srcdoc` iframe that works in Colab's network-restricted output frames.

### Viewer controls

Drag = rotate · wheel = zoom · right-drag = pan · hover = exact stats ·
click = inspect (+ diagnostics filter) · double-click = expand/collapse ·
**2D Ortho** = torchview-style top-down view · **▶ Replay** = forward/backward
replay · toggles for tensor geometry, ±Z overlays, ops, rolling, labels.
The log-scale ruler beside the model gives the quantitative `1e^x` scale.

---

## torchview parity

The trace captures at full detail (torchview's information is a strict subset
of the IR), module/function attributes are collected by default
(`collect_attributes=False` to disable), and torchview's native graphviz
output stays available at any granularity:

```python
run.export_dot("model.gv", depth=3, graph_dir="LR")   # DOT source
run.draw_graph(depth=2, roll=True, save_graph=True,
               filename="model", format="png")        # torchview pipeline
```

`draw_graph`/`export_dot` accept every `torchview.draw_graph` option
(`depth`, `hide_inner_tensors`, `hide_module_functions`, `roll`, `graph_dir`,
`strict`, ...) and reuse the exact inputs of the original trace; runtime
capture hooks are suspended for the auxiliary pass.

---

## Repository layout

```
torchspace/
  _tv_compat.py    # single audited import surface over torchview
  tracer.py        # non-invasive structural tracer (extension points A & B)
  runtime.py       # forward/backward hook capture on plain tensors
  stats.py         # compact tensor statistics (never raw tensors)
  ir.py            # IR builder: stable IDs, empty-pass contraction, edges
  diagnostics.py   # rule-based anomaly detection
  api.py           # public view()/trace()/SpaceRun (+ draw_graph/export_dot)
  export.py        # embeds IR into the self-contained viewer
  assets/viewer.html       # packaged viewer (generated — do not edit)
viewer/
  src.html                 # viewer source (edit this)
  three_r128_inline.html   # vendored three.js r128 (script-wrapped)
scripts/build_viewer.py    # src.html + three -> torchspace/assets/viewer.html
docs/                      # GitHub Pages live demo + README assets
build_demo.py              # end-to-end driver -> out/torchspace_viewer.html
tests/test_smoke.py
```

Developing the viewer: edit `viewer/src.html`, then run
`python scripts/build_viewer.py` to regenerate the packaged asset, and
`python build_demo.py` to rebuild the demo scene.

## References

- **[torchview](https://github.com/mert-kurttutan/torchview)** — Kurttutan, M.
  *torchview: visualize PyTorch models* (MIT). TorchSpace extends its tracer
  non-invasively: torchview solves structural tracing (modules, functions,
  tensors, hierarchy, recursion) via `RecorderTensor` and a patched
  `nn.Module.__call__`, and TorchSpace composes around those two points to add
  stable IDs, execution order, runtime metadata and statistics.
- **[torchinfo](https://github.com/TylerYep/torchinfo)** — tabular model
  summaries; the lineage torchview grew from.
- **[three.js](https://threejs.org/)** — the WebGL renderer bundled in the
  viewer (r128, MIT, © Three.js Authors).
- Sugiyama, K., Tagawa, S., Toda, M. (1981). *Methods for visual understanding
  of hierarchical system structures.* — the layered graph drawing approach the
  viewer's deterministic layout follows.
- Glorot, X., Bengio, Y. (2010). *Understanding the difficulty of training
  deep feedforward neural networks.* — the vanishing/exploding signal
  phenomena the ±Z encoding is designed to make visible.

---

## Creator

<p align="center">
  <a href="https://x.com/so_kariyama"><img src="https://img.shields.io/badge/X_(Twitter)-%40so__kariyama-0e1013?style=for-the-badge&logo=x&logoColor=white" alt="X: @so_kariyama" /></a>
</p>
<p align="center">
  <a href="https://x.com/so_kariyama"><img src="https://raw.githubusercontent.com/kariyamaso/torchspace/main/docs/assets/qr.png" width="150" alt="QR code for X @so_kariyama" /></a>
</p>

## License

[MIT](https://github.com/kariyamaso/torchspace/blob/main/LICENSE) — torchview
(MIT) is imported as a dependency, none of its code
is copied; three.js r128 (MIT, © Three.js Authors) is bundled inside the
packaged viewer with its license header retained.
