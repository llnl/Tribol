#!/usr/bin/env python3
"""Create a standalone HTML animation for EnergyMortar single-pair sweeps.

The input is a CSV with one row per sweep step. Required geometry columns are
the two endpoints for the non-mortar edge A and mortar edge B. Accepted names:

  A0_x,A0_y,A1_x,A1_y,B0_x,B0_y,B1_x,B1_y

Lower-case compact aliases such as a0x,a0y,a1x,a1y,b0x,b0y,b1x,b1y are also
accepted. The generated view shows the moving edge pair, an energy history, and
all available nodal force components.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Iterable


GEOM_ALIASES = {
    "a0x": ("A0_x", "a0_x", "a0x", "xA0", "ax0"),
    "a0y": ("A0_y", "a0_y", "a0y", "yA0", "ay0"),
    "a1x": ("A1_x", "a1_x", "a1x", "xA1", "ax1"),
    "a1y": ("A1_y", "a1_y", "a1y", "yA1", "ay1"),
    "b0x": ("B0_x", "b0_x", "b0x", "xB0", "bx0"),
    "b0y": ("B0_y", "b0_y", "b0y", "yB0", "by0"),
    "b1x": ("B1_x", "b1_x", "b1x", "xB1", "bx1"),
    "b1y": ("B1_y", "b1_y", "b1y", "yB1", "by1"),
}

FORCE_ALIASES = [
    (("fA0_x", "fa0x", "fx0"), ("fA0_y", "fa0y", "fy0")),
    (("fA1_x", "fa1x", "fx1"), ("fA1_y", "fa1y", "fy1")),
    (("fB0_x", "fb0x", "fx2"), ("fB0_y", "fb0y", "fy2")),
    (("fB1_x", "fb1x", "fx3"), ("fB1_y", "fb1y", "fy3")),
]

FORCE_COMPONENT_NAMES = ("A0x", "A0y", "A1x", "A1y", "B0x", "B0y", "B1x", "B1y")

DEFAULT_METRICS = (
    "energy",
    "normal_force",
    "force_normal",
    "force_dot_direction",
    "gap",
    "gtilde0",
    "gtilde1",
    "area0",
    "area1",
    "proj_min",
    "proj_max",
    "smooth_min",
    "smooth_max",
)


def first_float(row: dict[str, str], names: Iterable[str], default: float | None = None) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def require_float(row: dict[str, str], key: str) -> float:
    value = first_float(row, GEOM_ALIASES[key])
    if value is None:
        raise ValueError(f"missing required geometry column for {key}: one of {GEOM_ALIASES[key]}")
    return value


def metric_names(fieldnames: list[str], requested: str | None) -> list[str]:
    if requested:
        names = [item.strip() for item in requested.split(",") if item.strip()]
        return [name for name in names if name in fieldnames]
    return [name for name in DEFAULT_METRICS if name in fieldnames]


def load_frames(csv_path: Path, case: str | None, requested_metrics: str | None) -> tuple[list[dict], list[str]]:
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header")
        fields = reader.fieldnames
        metrics = metric_names(fields, requested_metrics)
        frames = []
        for index, row in enumerate(reader):
            if case is not None and row.get("case") != case:
                continue
            points = [
                [require_float(row, "a0x"), require_float(row, "a0y")],
                [require_float(row, "a1x"), require_float(row, "a1y")],
                [require_float(row, "b0x"), require_float(row, "b0y")],
                [require_float(row, "b1x"), require_float(row, "b1y")],
            ]
            forces = []
            has_force = False
            for xnames, ynames in FORCE_ALIASES:
                fx = first_float(row, xnames, 0.0)
                fy = first_float(row, ynames, 0.0)
                has_force = has_force or abs(fx or 0.0) > 0.0 or abs(fy or 0.0) > 0.0
                forces.append([fx or 0.0, fy or 0.0])
            frame_metrics = {
                name: first_float(row, (name,), None)
                for name in metrics
                if first_float(row, (name,), None) is not None
            }
            frames.append(
                {
                    "index": int(first_float(row, ("step",), index) or index),
                    "s": first_float(row, ("s", "time", "disp", "displacement"), float(index)),
                    "case": row.get("case", ""),
                    "points": points,
                    "forces": forces if has_force else [],
                    "metrics": frame_metrics,
                }
            )
    if not frames:
        raise ValueError("no frames matched the input CSV and case filter")
    return frames, metrics


def data_bounds(frames: list[dict]) -> dict[str, float]:
    xs = []
    ys = []
    for frame in frames:
        for x, y in frame["points"]:
            xs.append(x)
            ys.append(y)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    pad = 0.18 * span
    return {"xmin": xmin - pad, "xmax": xmax + pad, "ymin": ymin - pad, "ymax": ymax + pad}


def metric_ranges(frames: list[dict], metrics: list[str]) -> dict[str, list[float]]:
    ranges = {}
    for name in metrics:
        values = [frame["metrics"][name] for frame in frames if name in frame["metrics"]]
        if not values:
            continue
        vmin, vmax = min(values), max(values)
        if abs(vmax - vmin) < 1.0e-14:
            pad = max(abs(vmin) * 0.1, 1.0)
        else:
            pad = 0.08 * (vmax - vmin)
        ranges[name] = [vmin - pad, vmax + pad]
    return ranges


def write_html(frames: list[dict], metrics: list[str], title: str, output: Path, fps: float) -> None:
    payload = {
        "title": title,
        "fps": fps,
        "bounds": data_bounds(frames),
        "metrics": [name for name in metrics if any(name in frame["metrics"] for frame in frames)],
        "metricRanges": metric_ranges(frames, metrics),
        "forceComponentNames": FORCE_COMPONENT_NAMES,
        "frames": frames,
    }
    rendered = HTML_TEMPLATE.replace("__PAYLOAD_TITLE__", html.escape(title))
    rendered = rendered.replace("__PAYLOAD__", json.dumps(payload))
    output.write_text(rendered, encoding="utf-8")


def write_example(csv_path: Path) -> None:
    fields = [
        "case",
        "step",
        "s",
        "A0_x",
        "A0_y",
        "A1_x",
        "A1_y",
        "B0_x",
        "B0_y",
        "B1_x",
        "B1_y",
        "energy",
        "normal_force",
        "gap",
        "proj_min",
        "proj_max",
        "smooth_min",
        "smooth_max",
        "fx0",
        "fy0",
        "fx1",
        "fy1",
        "fx2",
        "fy2",
        "fx3",
        "fy3",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(90):
            t = step / 89.0
            slide = -0.75 + 1.5 * t
            gap = 0.18 - 0.34 * math.sin(math.pi * t)
            penetration = max(0.0, -gap)
            energy = 0.5 * 75.0 * penetration * penetration
            normal_force = 75.0 * penetration
            a0 = (-0.5, 0.0)
            a1 = (0.5, 0.0)
            b0 = (-0.42 + slide, gap)
            b1 = (0.42 + slide, gap + 0.06 * math.sin(2.0 * math.pi * t))
            writer.writerow(
                {
                    "case": "example_slide",
                    "step": step,
                    "s": f"{slide:.8g}",
                    "A0_x": a0[0],
                    "A0_y": a0[1],
                    "A1_x": a1[0],
                    "A1_y": a1[1],
                    "B0_x": b0[0],
                    "B0_y": b0[1],
                    "B1_x": b1[0],
                    "B1_y": b1[1],
                    "energy": f"{energy:.12g}",
                    "normal_force": f"{normal_force:.12g}",
                    "gap": f"{gap:.12g}",
                    "proj_min": f"{b0[0]:.12g}",
                    "proj_max": f"{b1[0]:.12g}",
                    "smooth_min": f"{max(-0.5, b0[0]):.12g}",
                    "smooth_max": f"{min(0.5, b1[0]):.12g}",
                    "fx0": 0.0,
                    "fy0": f"{-0.5 * normal_force:.12g}",
                    "fx1": 0.0,
                    "fy1": f"{-0.5 * normal_force:.12g}",
                    "fx2": 0.0,
                    "fy2": f"{0.5 * normal_force:.12g}",
                    "fx3": 0.0,
                    "fy3": f"{0.5 * normal_force:.12g}",
                }
            )


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAYLOAD_TITLE__</title>
<style>
:root {
  color-scheme: light;
  --ink: #1f2933;
  --muted: #667085;
  --grid: #d9dee7;
  --panel: #f6f8fb;
  --a: #1b75bb;
  --b: #c2410c;
  --force: #2f7d32;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: white;
}
.stage {
  width: min(1280px, 100vw);
  margin: 0 auto;
  padding: 22px 26px 18px;
}
.top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 10px;
}
h1 {
  font-size: 25px;
  line-height: 1.1;
  margin: 0;
  font-weight: 720;
}
.meta {
  color: var(--muted);
  font-size: 14px;
  white-space: nowrap;
}
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(380px, 0.9fr);
  gap: 18px;
  align-items: stretch;
}
.panel {
  border: 1px solid #d0d7e2;
  background: var(--panel);
  border-radius: 6px;
  overflow: hidden;
}
svg { display: block; width: 100%; height: auto; }
.controls {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  margin-top: 13px;
}
button {
  border: 1px solid #b7c0cc;
  background: #ffffff;
  color: var(--ink);
  border-radius: 5px;
  font-size: 14px;
  padding: 7px 12px;
  cursor: pointer;
}
input[type="range"] { width: 100%; }
.legend {
  display: flex;
  gap: 16px;
  align-items: center;
  color: var(--muted);
  font-size: 13px;
  margin-top: 8px;
}
.swatch {
  display: inline-block;
  width: 20px;
  height: 3px;
  margin-right: 6px;
  vertical-align: middle;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .meta { white-space: normal; }
}
</style>
</head>
<body>
<div class="stage">
  <div class="top">
    <h1 id="title"></h1>
    <div class="meta" id="meta"></div>
  </div>
  <div class="layout">
    <div>
      <div class="panel"><svg id="geom" viewBox="0 0 720 500" role="img"></svg></div>
      <div class="legend">
        <span><span class="swatch" style="background: var(--a)"></span>edge A</span>
        <span><span class="swatch" style="background: var(--b)"></span>edge B</span>
        <span><span class="swatch" style="background: var(--force)"></span>force</span>
      </div>
    </div>
    <div class="panel"><svg id="plots" viewBox="0 0 640 500" role="img"></svg></div>
  </div>
  <div class="controls">
    <button id="play">Pause</button>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <div class="meta" id="readout"></div>
  </div>
</div>
<script>
const payload = __PAYLOAD__;
document.title = payload.title;
document.getElementById("title").textContent = payload.title;

const geomSvg = document.getElementById("geom");
const plotSvg = document.getElementById("plots");
const playButton = document.getElementById("play");
const scrub = document.getElementById("scrub");
const readout = document.getElementById("readout");
const meta = document.getElementById("meta");

const frames = payload.frames;
const metrics = payload.metrics;
scrub.max = String(frames.length - 1);
meta.textContent = `${frames.length} frames at ${payload.fps} fps`;

let current = 0;
let playing = true;
let lastTime = 0;
const ns = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(ns, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.appendChild(node);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function sx(x) {
  const b = payload.bounds;
  return 60 + (x - b.xmin) / (b.xmax - b.xmin) * 600;
}

function sy(y) {
  const b = payload.bounds;
  return 440 - (y - b.ymin) / (b.ymax - b.ymin) * 380;
}

function metricX(i) {
  return 64 + i / Math.max(frames.length - 1, 1) * 430;
}

function paddedRange(values, fallback = [0, 1]) {
  if (!values.length) return fallback;
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (Math.abs(hi - lo) < 1e-14) {
    const pad = Math.max(Math.abs(lo) * 0.1, 1.0);
    lo -= pad;
    hi += pad;
  } else {
    const pad = 0.08 * (hi - lo);
    lo -= pad;
    hi += pad;
  }
  return [lo, hi];
}

function rangeY(range, value, top, height) {
  return top + height - (value - range[0]) / (range[1] - range[0]) * height;
}

function energyValues() {
  return frames.map(f => f.metrics.energy).filter(v => Number.isFinite(v));
}

function forceValues() {
  const values = [];
  frames.forEach(frame => {
    if (!frame.forces || !frame.forces.length) return;
    frame.forces.forEach(f => {
      values.push(f[0], f[1]);
    });
  });
  return values;
}

function forceComponent(frame, component) {
  if (!frame.forces || frame.forces.length < 4) return null;
  const node = Math.floor(component / 2);
  const dim = component % 2;
  return frame.forces[node][dim];
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) {
    return value.toExponential(2);
  }
  return value.toPrecision(3);
}

function drawAxis(top, height, title, range) {
  el("line", {x1: 64, y1: top + height, x2: 494, y2: top + height, stroke: "#9aa6b2", "stroke-width": 1.2}, plotSvg);
  el("line", {x1: 64, y1: top, x2: 64, y2: top + height, stroke: "#9aa6b2", "stroke-width": 1.2}, plotSvg);
  el("text", {x: 64, y: top - 12, fill: "#1f2933", "font-size": 16, "font-weight": 700}, plotSvg).textContent = title;
  el("text", {x: 504, y: top + 5, fill: "#667085", "font-size": 12}, plotSvg).textContent = formatNumber(range[1]);
  el("text", {x: 504, y: top + height, fill: "#667085", "font-size": 12}, plotSvg).textContent = formatNumber(range[0]);
  if (range[0] < 0 && range[1] > 0) {
    const y0 = rangeY(range, 0, top, height);
    el("line", {x1: 64, y1: y0, x2: 494, y2: y0, stroke: "#c8d0da", "stroke-width": 1, "stroke-dasharray": "4 4"}, plotSvg);
  }
}

function drawGeom(frame) {
  clear(geomSvg);
  el("rect", {x: 0, y: 0, width: 720, height: 500, fill: "#f6f8fb"}, geomSvg);
  for (let i = 0; i <= 6; ++i) {
    const x = 60 + i * 100;
    el("line", {x1: x, y1: 60, x2: x, y2: 440, stroke: "#d9dee7", "stroke-width": 1}, geomSvg);
  }
  for (let i = 0; i <= 4; ++i) {
    const y = 60 + i * 95;
    el("line", {x1: 60, y1: y, x2: 660, y2: y, stroke: "#d9dee7", "stroke-width": 1}, geomSvg);
  }

  const p = frame.points;
  const a0 = [sx(p[0][0]), sy(p[0][1])], a1 = [sx(p[1][0]), sy(p[1][1])];
  const b0 = [sx(p[2][0]), sy(p[2][1])], b1 = [sx(p[3][0]), sy(p[3][1])];

  el("line", {x1: a0[0], y1: a0[1], x2: a1[0], y2: a1[1], stroke: "#1b75bb", "stroke-width": 8, "stroke-linecap": "round"}, geomSvg);
  el("line", {x1: b0[0], y1: b0[1], x2: b1[0], y2: b1[1], stroke: "#c2410c", "stroke-width": 8, "stroke-linecap": "round"}, geomSvg);
  for (const pt of [a0, a1]) el("circle", {cx: pt[0], cy: pt[1], r: 7, fill: "#ffffff", stroke: "#1b75bb", "stroke-width": 3}, geomSvg);
  for (const pt of [b0, b1]) el("circle", {cx: pt[0], cy: pt[1], r: 7, fill: "#ffffff", stroke: "#c2410c", "stroke-width": 3}, geomSvg);

  if (frame.forces && frame.forces.length) {
    const points = [a0, a1, b0, b1];
    const maxMag = Math.max(...frame.forces.map(f => Math.hypot(f[0], f[1])), 1e-12);
    for (let i = 0; i < frame.forces.length; ++i) {
      const f = frame.forces[i];
      const mag = Math.hypot(f[0], f[1]);
      if (mag <= 0) continue;
      const scale = 58 / maxMag;
      const x1 = points[i][0], y1 = points[i][1];
      const x2 = x1 + f[0] * scale, y2 = y1 - f[1] * scale;
      el("line", {x1, y1, x2, y2, stroke: "#2f7d32", "stroke-width": 3, "marker-end": "url(#arrow)"}, geomSvg);
    }
  }

  const defs = el("defs", {}, geomSvg);
  const marker = el("marker", {id: "arrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse"}, defs);
  el("path", {d: "M 0 0 L 10 5 L 0 10 z", fill: "#2f7d32"}, marker);
}

function drawPlots() {
  clear(plotSvg);
  el("rect", {x: 0, y: 0, width: 640, height: 500, fill: "#f6f8fb"}, plotSvg);

  const energyRange = paddedRange(energyValues());
  const forceRange = paddedRange(forceValues());
  const energyTop = 48;
  const energyHeight = 125;
  const forceTop = 245;
  const forceHeight = 175;
  const colors = ["#1b75bb", "#73a7d5", "#c2410c", "#e19a72", "#2f7d32", "#84b982", "#7c3aed", "#b197fc"];

  drawAxis(energyTop, energyHeight, "Energy", energyRange);
  const hasEnergy = energyValues().length > 0;
  if (hasEnergy) {
    let d = "";
    frames.forEach((frame, i) => {
      if (!Number.isFinite(frame.metrics.energy)) return;
      const x = metricX(i);
      const y = rangeY(energyRange, frame.metrics.energy, energyTop, energyHeight);
      d += `${d ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)} `;
    });
    el("path", {d, fill: "none", stroke: "#111827", "stroke-width": 2.8}, plotSvg);
  } else {
    el("text", {x: 80, y: energyTop + 68, fill: "#667085", "font-size": 14}, plotSvg).textContent = "No energy column";
  }

  drawAxis(forceTop, forceHeight, "Nodal Force Components", forceRange);
  const hasForces = forceValues().length > 0;
  if (hasForces) {
    for (let component = 0; component < 8; ++component) {
      let d = "";
      frames.forEach((frame, i) => {
        const value = forceComponent(frame, component);
        if (!Number.isFinite(value)) return;
        const x = metricX(i);
        const y = rangeY(forceRange, value, forceTop, forceHeight);
        d += `${d ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)} `;
      });
      el("path", {
        d,
        fill: "none",
        stroke: colors[component],
        "stroke-width": 1.8,
        "stroke-opacity": 0.95
      }, plotSvg);
    }
  } else {
    el("text", {x: 80, y: forceTop + 90, fill: "#667085", "font-size": 14}, plotSvg).textContent = "No force columns";
  }

  const legendX = 516;
  const legendY = forceTop + 12;
  for (let component = 0; component < 8; ++component) {
    const y = legendY + component * 18;
    el("line", {x1: legendX, y1: y - 4, x2: legendX + 20, y2: y - 4, stroke: colors[component], "stroke-width": 2.4}, plotSvg);
    el("text", {x: legendX + 26, y, fill: "#344054", "font-size": 12}, plotSvg).textContent =
      payload.forceComponentNames[component];
  }
}

function drawMarker(frameIndex) {
  plotSvg.querySelectorAll(".marker").forEach(node => node.remove());
  const frame = frames[frameIndex];
  const x = metricX(frameIndex);
  el("line", {class: "marker", x1: x, y1: 34, x2: x, y2: 430, stroke: "#111827", "stroke-width": 1.2, "stroke-dasharray": "5 5"}, plotSvg);

  const energyRange = paddedRange(energyValues());
  if (Number.isFinite(frame.metrics.energy)) {
    const y = rangeY(energyRange, frame.metrics.energy, 48, 125);
    el("circle", {class: "marker", cx: x, cy: y, r: 4.8, fill: "#111827", stroke: "#ffffff", "stroke-width": 1.5}, plotSvg);
  }

  const forceRange = paddedRange(forceValues());
  for (let component = 0; component < 8; ++component) {
    const value = forceComponent(frame, component);
    if (!Number.isFinite(value)) continue;
    const y = rangeY(forceRange, value, 245, 175);
    el("circle", {class: "marker", cx: x, cy: y, r: 3.4, fill: "#111827", stroke: "#ffffff", "stroke-width": 1.0}, plotSvg);
  }
}

function render(index) {
  current = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[current];
  scrub.value = String(current);
  readout.textContent = `step ${frame.index}, s=${Number(frame.s).toPrecision(5)}`;
  drawGeom(frame);
  drawMarker(current);
}

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
});
scrub.addEventListener("input", () => {
  playing = false;
  playButton.textContent = "Play";
  render(Number(scrub.value));
});

drawPlots();
render(0);
function tick(time) {
  if (playing && (!lastTime || time - lastTime >= 1000 / payload.fps)) {
    render((current + 1) % frames.length);
    lastTime = time;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", help="input sweep CSV")
    parser.add_argument("-o", "--output", type=Path, default=Path("energy_mortar_single_pair.html"))
    parser.add_argument("--case", help="only animate rows with this case value")
    parser.add_argument("--metrics", help="comma-separated metric columns to plot")
    parser.add_argument("--title", default="EnergyMortar Single Element Pair")
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--write-example", type=Path, help="write an example CSV and exit")
    args = parser.parse_args()

    if args.write_example:
      write_example(args.write_example)
      return 0

    if args.csv is None:
        parser.error("csv is required unless --write-example is used")

    frames, metrics = load_frames(args.csv, args.case, args.metrics)
    write_html(frames, metrics, args.title, args.output, args.fps)
    print(f"wrote {args.output} with {len(frames)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
