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
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


GEOM_ALIASES = {
    "a0x": ("A0_x", "a0_x", "a0x", "xA0", "ax0"),
    "a0y": ("A0_y", "a0_y", "a0y", "yA0", "ay0"),
    "a1x": ("A1_x", "a1_x", "a1x", "xA1", "ax1"),
    "a1y": ("A1_y", "a1_y", "a1y", "yA1", "ay1"),
    "a2x": ("A2_x", "a2_x", "a2x", "xA2", "ax2"),
    "a2y": ("A2_y", "a2_y", "a2y", "yA2", "ay2"),
    "b0x": ("B0_x", "b0_x", "b0x", "xB0", "bx0"),
    "b0y": ("B0_y", "b0_y", "b0y", "yB0", "by0"),
    "b1x": ("B1_x", "b1_x", "b1x", "xB1", "bx1"),
    "b1y": ("B1_y", "b1_y", "b1y", "yB1", "by1"),
}

FORCE_ALIASES = [
    (("fA0_x", "fa0x", "fx0"), ("fA0_y", "fa0y", "fy0")),
    (("fA1_x", "fa1x", "fx1"), ("fA1_y", "fa1y", "fy1")),
    (("fA2_x", "fa2x"), ("fA2_y", "fa2y")),
    (("fB0_x", "fb0x", "fx2"), ("fB0_y", "fb0y", "fy2")),
    (("fB1_x", "fb1x", "fx3"), ("fB1_y", "fb1y", "fy3")),
]

FORCE_COMPONENT_NAMES = ("A0x", "A0y", "A1x", "A1y", "A2x", "A2y", "B0x", "B0y", "B1x", "B1y")
FORCE_COMPONENT_NAMES_SINGLE_A_EDGE = ("A0x", "A0y", "A1x", "A1y", "B0x", "B0y", "B1x", "B1y")

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
            a_points = [
                [require_float(row, "a0x"), require_float(row, "a0y")],
                [require_float(row, "a1x"), require_float(row, "a1y")],
            ]
            a2x = first_float(row, GEOM_ALIASES["a2x"], None)
            a2y = first_float(row, GEOM_ALIASES["a2y"], None)
            if a2x is not None and a2y is not None:
                a_points.append([a2x, a2y])
            b_points = [
                [require_float(row, "b0x"), require_float(row, "b0y")],
                [require_float(row, "b1x"), require_float(row, "b1y")],
            ]
            points = a_points + b_points
            force_aliases = FORCE_ALIASES if len(a_points) > 2 else [
                FORCE_ALIASES[0],
                FORCE_ALIASES[1],
                FORCE_ALIASES[3],
                FORCE_ALIASES[4],
            ]
            forces = []
            has_force_data = False
            for xnames, ynames in force_aliases:
                fx = first_float(row, xnames, None)
                fy = first_float(row, ynames, None)
                has_force_data = has_force_data or fx is not None or fy is not None
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
                    "aPoints": a_points,
                    "bPoints": b_points,
                    "forces": forces if has_force_data else [],
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
    force_component_names = (
        FORCE_COMPONENT_NAMES
        if any(len(frame.get("aPoints", [])) > 2 for frame in frames)
        else FORCE_COMPONENT_NAMES_SINGLE_A_EDGE
    )
    payload = {
        "title": title,
        "fps": fps,
        "bounds": data_bounds(frames),
        "metrics": [name for name in metrics if any(name in frame["metrics"] for frame in frames)],
        "metricRanges": metric_ranges(frames, metrics),
        "forceComponentNames": force_component_names,
        "frames": frames,
    }
    rendered = HTML_TEMPLATE.replace("__PAYLOAD_TITLE__", html.escape(title))
    rendered = rendered.replace("__PAYLOAD__", json.dumps(payload))
    output.write_text(rendered, encoding="utf-8")


def export_mp4(html_path: Path, frame_count: int, output: Path, fps: float, width: int, height: int) -> None:
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        missing.append("playwright")
        sync_playwright = None
        playwright_import_error = exc
    else:
        playwright_import_error = None

    if missing:
        details = []
        if "playwright" in missing:
            details.append("install Playwright with: python3 -m pip install playwright")
            details.append("install Chromium with: python3 -m playwright install chromium")
        if "ffmpeg" in missing:
            details.append("install ffmpeg and ensure it is on PATH")
        raise RuntimeError("Cannot export MP4; missing " + ", ".join(missing) + ". " + "; ".join(details)) from (
            playwright_import_error if "playwright" in missing else None
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="energy_mortar_mp4_") as tmp:
        frame_dir = Path(tmp)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
                page.goto(html_path.resolve().as_uri())
                page.wait_for_load_state("networkidle")
                page.evaluate("pauseAnimation()")
                page.evaluate("document.body.classList.add('export-mode')")
                for index in range(frame_count):
                    page.evaluate("(index) => render(index)", index)
                    page.screenshot(path=frame_dir / f"frame_{index:04d}.png")
            finally:
                browser.close()

        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)


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
  --ink: black;
  --muted: black;
  --grid: #a9aabc;
  --panel: white;
  --a: #3366CC;
  --b: #9d0c0c;
  --force: #84c342;
  --axis: #6e6e7c;
  --energy: #001E62;
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
  font-size: 33px;
  line-height: 1.1;
  margin: 0;
  font-weight: 720;
}
.meta {
  color: var(--muted);
  font-size: 18px;
  white-space: nowrap;
}
.layout {
  display: grid;
  grid-template-columns: minmax(320px, 2fr) minmax(560px, 3fr);
  gap: 18px;
  align-items: stretch;
}
.panel {
  background: var(--panel);
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
  border: 1px solid var(--grid);
  background: var(--panel);
  color: var(--ink);
  border-radius: 5px;
  font-size: 18px;
  padding: 7px 12px;
  cursor: pointer;
}
input[type="range"] { width: 100%; }
.legend {
  display: flex;
  gap: 16px;
  align-items: center;
  color: var(--muted);
  font-size: 17px;
  margin-top: 8px;
}
.swatch {
  display: inline-block;
  width: 20px;
  height: 3px;
  margin-right: 6px;
  vertical-align: middle;
}
body.export-mode .top,
body.export-mode .controls {
  display: none;
}
body.export-mode {
  overflow: hidden;
}
body.export-mode .stage {
  width: 1280px;
  margin: 0;
  transform: scale(1.5);
  transform-origin: top left;
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
    <div class="panel"><svg id="plots" viewBox="0 0 1020 500" role="img"></svg></div>
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
const colors = {
  impactBlue: "#0032a1",
  elementalNavy: "#001E62",
  energeticAzure: "#3366CC",
  livermoriumIce: "#eaf0fb",
  carbonGray: "#a9aabc",
  quantumSlate: "#6e6e7c",
  innovationYellow: "#fcb317",
  researchRed: "#9d0c0c",
  performancePink: "#b40f64",
  algorithmOrange: "#ff7900",
  solarYellow: "#ffd900",
  gammaGreen: "#84c342",
  extremeTurquoise: "#00a5b8",
  inspirationIndigo: "#4b0082"
};
const plotFrame = {left: 76, right: 700, labelX: 724};

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

function pathValue(frame, fallback) {
  const value = Number(frame ? frame.s : fallback);
  return Number.isFinite(value) ? value : fallback;
}

function pathRange() {
  const values = frames.map((frame, index) => pathValue(frame, index));
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || Math.abs(hi - lo) < 1e-14) {
    lo = 0;
    hi = Math.max(frames.length - 1, 1);
  }
  return [lo, hi];
}

function metricX(i) {
  const range = pathRange();
  const value = pathValue(frames[i], i);
  return plotFrame.left + (value - range[0]) / (range[1] - range[0]) * (plotFrame.right - plotFrame.left);
}

function linspace(min, max, count) {
  if (count <= 1) return [min];
  const values = [];
  for (let i = 0; i < count; ++i) values.push(min + (max - min) * i / (count - 1));
  return values;
}

function plotTickIndices(count = 5) {
  if (frames.length <= 1) return [0];
  const range = pathRange();
  const targets = linspace(range[0], range[1], Math.min(count, frames.length));
  const indices = targets.map(target => {
    let best = 0;
    let bestDistance = Infinity;
    frames.forEach((frame, index) => {
      const distance = Math.abs(pathValue(frame, index) - target);
      if (distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  });
  return [...new Set(indices)];
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

function maxForceMagnitude() {
  let maxMag = 0;
  frames.forEach(frame => {
    if (!frame.forces || !frame.forces.length) return;
    frame.forces.forEach(f => {
      maxMag = Math.max(maxMag, Math.hypot(f[0], f[1]));
    });
  });
  return Math.max(maxMag, 1e-12);
}

function forceComponent(frame, component) {
  if (!frame.forces || frame.forces.length < 4) return null;
  const node = Math.floor(component / 2);
  const dim = component % 2;
  if (node >= frame.forces.length) return null;
  return frame.forces[node][dim];
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) {
    return value.toExponential(2);
  }
  return value.toPrecision(3);
}

function formatAxisNumber(value) {
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) {
    return value.toExponential(1);
  }
  return Number(value.toPrecision(3)).toString();
}

function drawAxis(top, height, title, range, yLabel) {
  el("line", {x1: plotFrame.left, y1: top + height, x2: plotFrame.right, y2: top + height, stroke: colors.quantumSlate, "stroke-width": 1.2}, plotSvg);
  el("line", {x1: plotFrame.left, y1: top, x2: plotFrame.left, y2: top + height, stroke: colors.quantumSlate, "stroke-width": 1.2}, plotSvg);
  el("text", {x: plotFrame.left, y: top - 12, fill: "black", "font-size": 21, "font-weight": 700}, plotSvg).textContent = title;
  el("text", {x: plotFrame.labelX, y: top + 5, fill: "black", "font-size": 16}, plotSvg).textContent = formatNumber(range[1]);
  el("text", {x: plotFrame.labelX, y: top + height, fill: "black", "font-size": 16}, plotSvg).textContent = formatNumber(range[0]);
  el("text", {
    x: (plotFrame.left + plotFrame.right) * 0.5,
    y: top + height + 38,
    fill: "black",
    "font-size": 17,
    "text-anchor": "middle"
  }, plotSvg).textContent = "s";
  el("text", {
    x: 24,
    y: top + 0.5 * height,
    fill: "black",
    "font-size": 17,
    "text-anchor": "middle",
    transform: `rotate(-90 24 ${top + 0.5 * height})`
  }, plotSvg).textContent = yLabel;
  for (const index of plotTickIndices()) {
    const x = metricX(index);
    const s = frames[index] ? Number(frames[index].s) : index;
    el("line", {x1: x, y1: top + height, x2: x, y2: top + height + 6, stroke: colors.quantumSlate, "stroke-width": 1.1}, plotSvg);
    el("line", {x1: x, y1: top, x2: x, y2: top + height, stroke: colors.carbonGray, "stroke-width": 0.8}, plotSvg);
    el("text", {
      x,
      y: top + height + 20,
      fill: "black",
      "font-size": 14,
      "text-anchor": "middle"
    }, plotSvg).textContent = formatAxisNumber(s);
  }
  if (range[0] < 0 && range[1] > 0) {
    const y0 = rangeY(range, 0, top, height);
    el("line", {x1: plotFrame.left, y1: y0, x2: plotFrame.right, y2: y0, stroke: colors.carbonGray, "stroke-width": 1, "stroke-dasharray": "4 4"}, plotSvg);
  }
}

function drawGeomAxes() {
  const b = payload.bounds;
  const xTicks = linspace(b.xmin, b.xmax, 5);
  const yTicks = linspace(b.ymin, b.ymax, 5);
  for (const value of xTicks) {
    const x = sx(value);
    el("line", {x1: x, y1: 440, x2: x, y2: 446, stroke: colors.quantumSlate, "stroke-width": 1.1}, geomSvg);
    el("text", {
      x,
      y: 463,
      fill: "black",
      "font-size": 16,
      "text-anchor": "middle"
    }, geomSvg).textContent = formatAxisNumber(value);
  }
  for (const value of yTicks) {
    const y = sy(value);
    el("line", {x1: 60, y1: y, x2: 66, y2: y, stroke: colors.quantumSlate, "stroke-width": 1.1}, geomSvg);
    el("text", {
      x: 70,
      y: y + 4,
      fill: "black",
      "font-size": 16,
      "text-anchor": "start"
    }, geomSvg).textContent = formatAxisNumber(value);
  }
  el("line", {x1: 60, y1: 440, x2: 660, y2: 440, stroke: colors.quantumSlate, "stroke-width": 1.2}, geomSvg);
  el("line", {x1: 60, y1: 60, x2: 60, y2: 440, stroke: colors.quantumSlate, "stroke-width": 1.2}, geomSvg);
  el("text", {x: 360, y: 490, fill: "black", "font-size": 18, "text-anchor": "middle"}, geomSvg).textContent = "x";
  el("text", {
    x: 18,
    y: 250,
    fill: "black",
    "font-size": 18,
    "text-anchor": "middle",
    transform: "rotate(-90 18 250)"
  }, geomSvg).textContent = "y";
}

function drawGeom(frame) {
  clear(geomSvg);
  el("rect", {x: 0, y: 0, width: 720, height: 500, fill: "white"}, geomSvg);
  for (let i = 0; i <= 6; ++i) {
    const x = 60 + i * 100;
    el("line", {x1: x, y1: 60, x2: x, y2: 440, stroke: colors.carbonGray, "stroke-width": 1}, geomSvg);
  }
  for (let i = 0; i <= 4; ++i) {
    const y = 60 + i * 95;
    el("line", {x1: 60, y1: y, x2: 660, y2: y, stroke: colors.carbonGray, "stroke-width": 1}, geomSvg);
  }
  drawGeomAxes();

  const aPts = (frame.aPoints || frame.points.slice(0, 2)).map(p => [sx(p[0]), sy(p[1])]);
  const bPts = (frame.bPoints || frame.points.slice(2, 4)).map(p => [sx(p[0]), sy(p[1])]);
  const a0 = aPts[0], a1 = aPts[1];
  const b0 = bPts[0], b1 = bPts[1];

  for (let i = 0; i + 1 < aPts.length; ++i) {
    el("line", {
      x1: aPts[i][0],
      y1: aPts[i][1],
      x2: aPts[i + 1][0],
      y2: aPts[i + 1][1],
      stroke: colors.energeticAzure,
      "stroke-width": 8,
      "stroke-linecap": "round"
    }, geomSvg);
  }
  el("line", {x1: b0[0], y1: b0[1], x2: b1[0], y2: b1[1], stroke: colors.researchRed, "stroke-width": 8, "stroke-linecap": "round"}, geomSvg);
  for (const pt of aPts) el("circle", {cx: pt[0], cy: pt[1], r: 7, fill: "white", stroke: colors.energeticAzure, "stroke-width": 3}, geomSvg);
  for (const pt of bPts) el("circle", {cx: pt[0], cy: pt[1], r: 7, fill: "white", stroke: colors.researchRed, "stroke-width": 3}, geomSvg);

  if (frame.forces && frame.forces.length) {
    const points = aPts.concat(bPts);
    const maxMag = maxForceMagnitude();
    for (let i = 0; i < Math.min(frame.forces.length, points.length); ++i) {
      const f = frame.forces[i];
      const mag = Math.hypot(f[0], f[1]);
      if (mag <= 0) continue;
      const scale = 58 / maxMag;
      const x1 = points[i][0], y1 = points[i][1];
      const x2 = x1 - f[0] * scale, y2 = y1 + f[1] * scale;
      el("line", {x1, y1, x2, y2, stroke: colors.gammaGreen, "stroke-width": 3, "marker-end": "url(#arrow)"}, geomSvg);
    }
  }

  const defs = el("defs", {}, geomSvg);
  const marker = el("marker", {id: "arrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse"}, defs);
  el("path", {d: "M 0 0 L 10 5 L 0 10 z", fill: colors.gammaGreen}, marker);
}

function drawPlots() {
  clear(plotSvg);
  el("rect", {x: 0, y: 0, width: 1020, height: 500, fill: "white"}, plotSvg);

  const energyRange = paddedRange(energyValues());
  const forceRange = paddedRange(forceValues());
  const energyTop = 48;
  const energyHeight = 125;
  const forceTop = 260;
  const forceHeight = 160;
  const forceColors = [
    colors.impactBlue,
    colors.energeticAzure,
    colors.extremeTurquoise,
    colors.gammaGreen,
    colors.researchRed,
    colors.performancePink,
    colors.algorithmOrange,
    colors.solarYellow,
    colors.inspirationIndigo,
    colors.quantumSlate
  ];

  drawAxis(energyTop, energyHeight, "Energy", energyRange, "energy");
  const hasEnergy = energyValues().length > 0;
  if (hasEnergy) {
    let d = "";
    frames.forEach((frame, i) => {
      if (!Number.isFinite(frame.metrics.energy)) return;
      const x = metricX(i);
      const y = rangeY(energyRange, frame.metrics.energy, energyTop, energyHeight);
      d += `${d ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)} `;
    });
    el("path", {d, fill: "none", stroke: colors.elementalNavy, "stroke-width": 2.8}, plotSvg);
  } else {
    el("text", {x: 92, y: energyTop + 68, fill: "black", "font-size": 18}, plotSvg).textContent = "No energy column";
  }

  drawAxis(forceTop, forceHeight, "Nodal Force Components", forceRange, "force");
  const hasForces = forceValues().length > 0;
  if (hasForces) {
    for (let component = 0; component < payload.forceComponentNames.length; ++component) {
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
        stroke: forceColors[component],
        "stroke-width": 1.8,
        "stroke-opacity": 0.95
      }, plotSvg);
    }
  } else {
    el("text", {x: 92, y: forceTop + 90, fill: "black", "font-size": 18}, plotSvg).textContent = "No force columns";
  }

  const legendX = 820;
  const legendY = forceTop + 12;
  for (let component = 0; component < payload.forceComponentNames.length; ++component) {
    const y = legendY + component * 18;
    el("line", {x1: legendX, y1: y - 4, x2: legendX + 20, y2: y - 4, stroke: forceColors[component], "stroke-width": 2.4}, plotSvg);
    el("text", {x: legendX + 26, y, fill: "black", "font-size": 16}, plotSvg).textContent =
      payload.forceComponentNames[component];
  }
}

function drawMarker(frameIndex) {
  plotSvg.querySelectorAll(".marker").forEach(node => node.remove());
  const frame = frames[frameIndex];
  const x = metricX(frameIndex);
  el("line", {class: "marker", x1: x, y1: 34, x2: x, y2: 430, stroke: colors.elementalNavy, "stroke-width": 1.2, "stroke-dasharray": "5 5"}, plotSvg);

  const energyRange = paddedRange(energyValues());
  if (Number.isFinite(frame.metrics.energy)) {
    const y = rangeY(energyRange, frame.metrics.energy, 48, 125);
    el("circle", {class: "marker", cx: x, cy: y, r: 4.8, fill: colors.elementalNavy, stroke: "white", "stroke-width": 1.5}, plotSvg);
  }

  const forceRange = paddedRange(forceValues());
  for (let component = 0; component < payload.forceComponentNames.length; ++component) {
    const value = forceComponent(frame, component);
    if (!Number.isFinite(value)) continue;
    const y = rangeY(forceRange, value, 260, 160);
    el("circle", {class: "marker", cx: x, cy: y, r: 3.4, fill: colors.elementalNavy, stroke: "white", "stroke-width": 1.0}, plotSvg);
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

function pauseAnimation() {
  playing = false;
  playButton.textContent = "Play";
}

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
});
scrub.addEventListener("input", () => {
  pauseAnimation();
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
    parser.add_argument("--export-mp4", type=Path, help="render the animation to an MP4 file")
    parser.add_argument("--export-duration", type=float, default=10.0, help="target MP4 duration in seconds")
    parser.add_argument("--export-width", type=int, default=1920, help="MP4 capture viewport width in pixels")
    parser.add_argument("--export-height", type=int, default=1080, help="MP4 capture viewport height in pixels")
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
    if args.export_mp4:
        if args.export_duration <= 0.0:
            parser.error("--export-duration must be positive")
        export_fps = len(frames) / args.export_duration
        try:
            export_mp4(args.output, len(frames), args.export_mp4, export_fps, args.export_width, args.export_height)
        except RuntimeError as exc:
            parser.error(str(exc))
        print(
            f"wrote {args.export_mp4} with {len(frames)} frames at {export_fps:.6g} fps "
            f"for {args.export_duration:.6g} seconds"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
