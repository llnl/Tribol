#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    case: str
    size: int
    warmups: int
    iterations: int
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class Mismatch:
    path: str
    current: Any
    reference: Any
    absolute_error: float | None
    tolerance: float | None
    message: str


@dataclass(frozen=True)
class TimingSummary:
    minimum: float
    median: float
    mean: float
    maximum: float
    standard_deviation: float


@dataclass(frozen=True)
class Comparison:
    spec: RunSpec
    current: dict[str, Any]
    reference: dict[str, Any]
    mismatches: tuple[Mismatch, ...]
    current_timing: TimingSummary
    reference_timing: TimingSummary

    @property
    def exact(self) -> bool:
        return not self.mismatches

    @property
    def median_ratio(self) -> float:
        if self.reference_timing.median == 0.0:
            return math.inf
        return self.current_timing.median / self.reference_timing.median


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported suite manifest schema in {path}")
    if not isinstance(manifest.get("cases"), dict) or not isinstance(manifest.get("suites"), dict):
        raise ValueError(f"suite manifest {path} must contain cases and suites objects")
    return manifest


def _suite_entries(manifest: dict[str, Any], suite: str, active: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    if suite in active:
        raise ValueError(f"suite inclusion cycle: {' -> '.join(active + (suite,))}")
    suites = manifest["suites"]
    if suite not in suites:
        raise ValueError(f"unknown benchmark suite: {suite}")
    definition = suites[suite]
    if isinstance(definition, list):
        return definition
    if not isinstance(definition, dict):
        raise ValueError(f"suite {suite} must be a list or object")
    entries: list[dict[str, Any]] = []
    for included in definition.get("include", []):
        entries.extend(_suite_entries(manifest, included, active + (suite,)))
    entries.extend(definition.get("runs", []))
    return entries


def expand_suite(manifest: dict[str, Any], suite: str) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seen: set[tuple[str, int, int, int]] = set()
    for entry in _suite_entries(manifest, suite):
        case_name = entry.get("case")
        case = manifest["cases"].get(case_name)
        if not isinstance(case, dict):
            raise ValueError(f"suite {suite} references unknown case: {case_name}")
        spec = RunSpec(
            case=case_name,
            size=int(entry["size"]),
            warmups=int(entry.get("warmups", case.get("warmups", 2))),
            iterations=int(entry.get("iterations", case.get("iterations", 10))),
            absolute_tolerance=float(entry.get("absolute_tolerance", case["absolute_tolerance"])),
            relative_tolerance=float(entry.get("relative_tolerance", case["relative_tolerance"])),
        )
        if spec.size <= 0 or spec.warmups < 0 or spec.iterations <= 0:
            raise ValueError(f"invalid run counts for {spec.case}")
        identity = (spec.case, spec.size, spec.warmups, spec.iterations)
        if identity not in seen:
            specs.append(spec)
            seen.add(identity)
    return specs


def parse_result(text: str, expected: RunSpec | None = None) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{"):
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                payload = loaded
                break
    if payload is None:
        raise ValueError("benchmark executable did not emit a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("benchmark result uses an unsupported schema")
    if expected and (payload.get("case") != expected.case or payload.get("size") != expected.size):
        raise ValueError("benchmark result does not match the requested case and size")
    timings = payload.get("timings", {}).get("contact_step_seconds")
    if not isinstance(timings, list) or not timings or not all(_is_number(value) for value in timings):
        raise ValueError("benchmark result has invalid timing samples")
    exactness = payload.get("exactness")
    if not isinstance(exactness, dict) or not isinstance(exactness.get("scalars"), dict) or not isinstance(
        exactness.get("vectors"), dict
    ):
        raise ValueError("benchmark result has invalid exactness data")
    return payload


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def summarize_timing(samples: list[float]) -> TimingSummary:
    values = [float(value) for value in samples]
    return TimingSummary(
        minimum=min(values),
        median=statistics.median(values),
        mean=statistics.fmean(values),
        maximum=max(values),
        standard_deviation=statistics.pstdev(values),
    )


def _compare_number(path: str, current: Any, reference: Any, spec: RunSpec) -> Mismatch | None:
    if not _is_number(current) or not _is_number(reference):
        return Mismatch(path, current, reference, None, None, "value is not a finite number")
    current_value = float(current)
    reference_value = float(reference)
    error = abs(current_value - reference_value)
    tolerance = spec.absolute_tolerance + spec.relative_tolerance * max(abs(current_value), abs(reference_value))
    if error <= tolerance:
        return None
    return Mismatch(path, current, reference, error, tolerance, "values differ beyond tolerance")


def compare_results(spec: RunSpec, current: dict[str, Any], reference: dict[str, Any]) -> Comparison:
    mismatches: list[Mismatch] = []
    current_exactness = current["exactness"]
    reference_exactness = reference["exactness"]
    for group in ("scalars", "vectors"):
        current_group = current_exactness[group]
        reference_group = reference_exactness[group]
        for name in sorted(set(current_group) | set(reference_group)):
            path = f"exactness.{group}.{name}"
            if name not in current_group or name not in reference_group:
                mismatches.append(
                    Mismatch(path, current_group.get(name), reference_group.get(name), None, None, "metric is missing")
                )
                continue
            if group == "scalars":
                mismatch = _compare_number(path, current_group[name], reference_group[name], spec)
                if mismatch:
                    mismatches.append(mismatch)
                continue
            current_values = current_group[name]
            reference_values = reference_group[name]
            if not isinstance(current_values, list) or not isinstance(reference_values, list):
                mismatches.append(
                    Mismatch(path, current_values, reference_values, None, None, "metric is not a vector")
                )
                continue
            if len(current_values) != len(reference_values):
                mismatches.append(
                    Mismatch(path, len(current_values), len(reference_values), None, None, "vector lengths differ")
                )
                continue
            for index, (current_value, reference_value) in enumerate(zip(current_values, reference_values)):
                mismatch = _compare_number(f"{path}[{index}]", current_value, reference_value, spec)
                if mismatch:
                    mismatches.append(mismatch)
    return Comparison(
        spec=spec,
        current=current,
        reference=reference,
        mismatches=tuple(mismatches),
        current_timing=summarize_timing(current["timings"]["contact_step_seconds"]),
        reference_timing=summarize_timing(reference["timings"]["contact_step_seconds"]),
    )


def report_data(comparisons: list[Comparison], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "exact": all(comparison.exact for comparison in comparisons),
        "comparisons": [
            {
                "spec": asdict(comparison.spec),
                "exact": comparison.exact,
                "mismatches": [asdict(mismatch) for mismatch in comparison.mismatches],
                "timing": {
                    "current": asdict(comparison.current_timing),
                    "reference": asdict(comparison.reference_timing),
                    "current_over_reference_median": comparison.median_ratio,
                },
                "results": {"current": comparison.current, "reference": comparison.reference},
            }
            for comparison in comparisons
        ],
    }
    if provenance is not None:
        report["provenance"] = provenance
    return report


def human_report(comparisons: list[Comparison]) -> str:
    lines = ["Tribol benchmark comparison", ""]
    for comparison in comparisons:
        status = "PASS" if comparison.exact else "FAIL"
        lines.append(
            f"{status:4} {comparison.spec.case:22} size={comparison.spec.size:<6} "
            f"new={comparison.current_timing.median:.6g}s ref={comparison.reference_timing.median:.6g}s "
            f"ratio={comparison.median_ratio:.3f}x"
        )
        for mismatch in comparison.mismatches[:10]:
            lines.append(
                f"     {mismatch.path}: current={mismatch.current!r} reference={mismatch.reference!r} "
                f"({mismatch.message})"
            )
        if len(comparison.mismatches) > 10:
            lines.append(f"     ... {len(comparison.mismatches) - 10} additional mismatches in JSON report")
    passed = sum(comparison.exact for comparison in comparisons)
    lines.extend(("", f"Exactness: {passed}/{len(comparisons)} runs passed. Timing ratios are informational."))
    return "\n".join(lines)
