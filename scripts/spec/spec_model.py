#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIREMENTS = REPO_ROOT / "src/tests/spec/requirements.yaml"
DEFAULT_CAPABILITIES = REPO_ROOT / "src/tests/spec/capabilities.yaml"


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class SpecFiles:
    requirements: Path = DEFAULT_REQUIREMENTS
    capabilities: Path = DEFAULT_CAPABILITIES


def load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpecError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise SpecError(f"{path} must contain an object")
    return document


def _require_unique_ids(items: Any, path: str) -> set[str]:
    if not isinstance(items, list) or not items:
        raise SpecError(f"{path} must be a nonempty list")
    identifiers: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise SpecError(f"{path}[{index}] must have a string id")
        identifier = item["id"]
        if identifier in identifiers:
            raise SpecError(f"duplicate id {identifier!r} in {path}")
        identifiers.add(identifier)
    return identifiers


POLICY_VALUES = {
    "geometry": {"ProjectedOverlap", "ConformingOverlap"},
    "normal": {"MeanPlane", "MortarSurface"},
    "integration": {"Centroid", "Polygon", "Face"},
    "constraint": {"Pointwise", "Nodal", "QuadraturePoint"},
    "basis": {"Primal", "Dual"},
    "enforcement": {"Penalty", "LagrangeMultiplier", "ExternalPressure", "None"},
    "stiffness": {"Constant", "Material"},
    "rate": {"None", "Constant", "Percentage"},
    "response": {"Frictionless", "ViscousTangential", "TiedNormal", "TiedFull"},
    "formulation": {"PointwiseTraction", "WeightedWeakForm", "Variational", "DiagnosticWeights"},
    "linearization": {"Exact", "Analytic", "Enzyme"},
}

METHOD_AXES = ("geometry", "integration", "constraint", "enforcement", "response", "formulation", "linearization")
VARIANT_AXES = {"response", "formulation", "linearization"}
DIMENSION_VALUES = {2, 3}
TOPOLOGY_VALUES = {"Segment", "Triangle", "Quadrilateral", "TessellatedHighOrderSegment"}
SEARCH_VALUES = {"CartesianProduct", "Grid", "Bvh"}
EXECUTION_VALUES = {"Sequential", "Deterministic", "Cuda"}
PARALLEL_VALUES = {"Serial", "MPI"}
OUTPUT_VALUES = {
    "Force",
    "PrimalForce",
    "ConstraintResidual",
    "Gap",
    "WeightedGap",
    "TributaryArea",
    "MortarWeights",
    "QuadratureGap",
    "QuadraturePressure",
    "Energy",
    "TimestepVote",
    "DerivativeOperators",
    "DerivativeBlocks",
}
REQUIREMENT_SCOPES = {
    "api",
    "core",
    "component",
    "derivative",
    "search",
    "adapter",
    "parallel",
    "conformance",
    "output",
    "quality",
}


def _require_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise SpecError(f"{path} must define exactly {', '.join(sorted(expected))}")


def _require_unique_values(value: Any, path: str, allowed: set[Any]) -> None:
    if not isinstance(value, list) or not value:
        raise SpecError(f"{path} must be a nonempty list")
    seen: set[Any] = set()
    for entry in value:
        if isinstance(entry, bool) or entry not in allowed:
            raise SpecError(f"{path} contains invalid value {entry!r}")
        if entry in seen:
            raise SpecError(f"{path} contains duplicate value {entry!r}")
        seen.add(entry)


def _validate_policy(axis: str, value: Any, combination_id: str) -> None:
    if not isinstance(value, dict) or value.get("policy") not in POLICY_VALUES[axis]:
        raise SpecError(f"{combination_id}: invalid {axis} policy")
    policy = value["policy"]
    path = f"{combination_id}: {axis}"
    if axis == "geometry" and policy == "ProjectedOverlap":
        _require_exact_keys(value, {"policy", "normal"}, path)
        if value["normal"] not in POLICY_VALUES["normal"]:
            raise SpecError(f"{combination_id}: ProjectedOverlap requires a valid normal")
    elif axis == "integration" and policy in {"Polygon", "Face"}:
        _require_exact_keys(value, {"policy", "order"}, path)
        if isinstance(value["order"], bool) or value["order"] not in {1, 2}:
            raise SpecError(f"{combination_id}: {policy} supports only orders 1 and 2")
    elif axis == "constraint" and policy == "Nodal":
        _require_exact_keys(value, {"policy", "basis"}, path)
        if value["basis"] not in POLICY_VALUES["basis"]:
            raise SpecError(f"{combination_id}: Nodal requires a valid basis")
    elif axis == "enforcement" and policy == "Penalty":
        _require_exact_keys(value, {"policy", "stiffness", "rate"}, path)
        if value.get("stiffness") not in POLICY_VALUES["stiffness"]:
            raise SpecError(f"{combination_id}: Penalty requires a valid stiffness")
        if value.get("rate") not in POLICY_VALUES["rate"]:
            raise SpecError(f"{combination_id}: Penalty requires a valid rate")
    else:
        _require_exact_keys(value, {"policy"}, path)


def _validate_variants(combination: dict[str, Any]) -> None:
    variants = combination.get("variants")
    if variants is None:
        return
    combination_id = combination["id"]
    if not isinstance(variants, dict) or not variants:
        raise SpecError(f"{combination_id}: variants must be a nonempty object")
    unknown_axes = set(variants) - VARIANT_AXES
    if unknown_axes:
        raise SpecError(f"{combination_id}: unsupported variant axes {sorted(unknown_axes)}")
    for axis, values in variants.items():
        _require_unique_values(values, f"{combination_id}: variants.{axis}", POLICY_VALUES[axis])
        if combination["method"][axis]["policy"] not in values:
            raise SpecError(f"{combination_id}: variants.{axis} must include the primary policy")


def method_variants(combination: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    variants = combination.get("variants", {})
    if not variants:
        yield combination["id"], combination["method"]
        return
    axes = sorted(variants)
    primary = combination["method"]
    for values in product(*(variants[axis] for axis in axes)):
        method = {axis: dict(value) for axis, value in primary.items()}
        suffix: list[str] = []
        for axis, value in zip(axes, values):
            method[axis] = {"policy": value}
            if value != primary[axis]["policy"]:
                suffix.extend((axis, value))
        identifier = combination["id"] if not suffix else "-".join((combination["id"], *suffix))
        yield identifier, method


def validate(files: SpecFiles) -> tuple[dict[str, Any], dict[str, Any]]:
    requirements = load_document(files.requirements)
    capabilities = load_document(files.capabilities)
    if requirements.get("schema_version") != 1 or capabilities.get("schema_version") != 1:
        raise SpecError("unsupported schema_version")

    requirement_items = requirements.get("requirements")
    requirement_ids = _require_unique_ids(requirement_items, "requirements")
    assert isinstance(requirement_items, list)
    for index, requirement in enumerate(requirement_items):
        _require_exact_keys(requirement, {"id", "scope", "text"}, f"requirements[{index}]")
        if not re.fullmatch(r"[A-Z]+-\d{3}", requirement["id"]):
            raise SpecError(f"requirements[{index}] has invalid id {requirement['id']!r}")
        if requirement["scope"] not in REQUIREMENT_SCOPES:
            raise SpecError(f"{requirement['id']}: invalid requirement scope")
        if not isinstance(requirement["text"], str) or not requirement["text"].strip():
            raise SpecError(f"{requirement['id']}: requirement text must be nonempty")
    combinations = capabilities.get("combinations")
    _require_unique_ids(combinations, "combinations")
    assert isinstance(combinations, list)

    seen_methods: dict[str, str] = {}
    for combination in combinations:
        combination_id = combination["id"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", combination_id):
            raise SpecError(f"invalid combination id {combination_id!r}")
        method = combination.get("method")
        if not isinstance(method, dict) or set(method) != set(METHOD_AXES):
            raise SpecError(f"{combination_id}: method must define exactly {', '.join(METHOD_AXES)}")
        for axis in METHOD_AXES:
            _validate_policy(axis, method[axis], combination_id)
        _validate_variants(combination)
        local_methods: set[str] = set()
        for variant_id, variant in method_variants(combination):
            for axis in METHOD_AXES:
                _validate_policy(axis, variant[axis], variant_id)
            signature = json.dumps(variant, sort_keys=True, separators=(",", ":"))
            if signature in local_methods:
                continue
            if signature in seen_methods:
                raise SpecError(f"{variant_id}: duplicates method advertised by {seen_methods[signature]}")
            local_methods.add(signature)
            seen_methods[signature] = variant_id
        references = combination.get("requirements")
        if not isinstance(references, list) or not references:
            raise SpecError(f"{combination_id}: requirements must be a nonempty list")
        if len(references) != len(set(references)):
            raise SpecError(f"{combination_id}: requirements contains duplicate values")
        unknown = set(references) - requirement_ids
        if unknown:
            raise SpecError(f"{combination_id}: unknown requirements {sorted(unknown)}")
        _require_unique_values(combination.get("dimensions"), f"{combination_id}: dimensions", DIMENSION_VALUES)
        _require_unique_values(combination.get("topologies"), f"{combination_id}: topologies", TOPOLOGY_VALUES)
        _require_unique_values(combination.get("search"), f"{combination_id}: search", SEARCH_VALUES)
        _require_unique_values(combination.get("execution"), f"{combination_id}: execution", EXECUTION_VALUES)
        _require_unique_values(combination.get("parallel"), f"{combination_id}: parallel", PARALLEL_VALUES)
        _require_unique_values(combination.get("outputs"), f"{combination_id}: outputs", OUTPUT_VALUES)
    return requirements, capabilities


def _cpp_geometry(value: dict[str, Any]) -> str:
    if value["policy"] == "ConformingOverlap":
        return "tribol::geometry::ConformingOverlap"
    return f"tribol::geometry::ProjectedOverlap<tribol::normal::{value['normal']}>"


def _cpp_ordered(namespace: str, value: dict[str, Any]) -> str:
    if "order" in value:
        return f"tribol::{namespace}::{value['policy']}<{value['order']}>"
    return f"tribol::{namespace}::{value['policy']}"


def _cpp_constraint(value: dict[str, Any]) -> str:
    if value["policy"] == "Nodal":
        return f"tribol::constraint::Nodal<tribol::basis::{value['basis']}>"
    return f"tribol::constraint::{value['policy']}"


def _cpp_enforcement(value: dict[str, Any]) -> str:
    if value["policy"] == "Penalty":
        return (
            f"tribol::enforcement::Penalty<tribol::stiffness::{value['stiffness']}, "
            f"tribol::rate::{value['rate']}>"
        )
    return f"tribol::enforcement::{value['policy']}"


def cpp_method(method: dict[str, Any]) -> str:
    arguments = [
        _cpp_geometry(method["geometry"]),
        _cpp_ordered("integration", method["integration"]),
        _cpp_constraint(method["constraint"]),
        _cpp_enforcement(method["enforcement"]),
        f"tribol::response::{method['response']['policy']}",
        f"tribol::formulation::{method['formulation']['policy']}",
        f"tribol::linearization::{method['linearization']['policy']}",
    ]
    return "tribol::Method<\n    " + ",\n    ".join(arguments) + ">"


def cpp_identifier(identifier: str) -> str:
    return "Spec_" + "_".join(part[:1].upper() + part[1:] for part in identifier.split("-"))


def emit_cpp(capabilities: dict[str, Any]) -> str:
    lines = ["// Generated by scripts/spec/validate.py --emit-cpp", "#include \"tribol/Tribol.hpp\"", ""]
    for combination in capabilities["combinations"]:
        for variant_id, method in method_variants(combination):
            name = cpp_identifier(variant_id)
            lines.extend(
                [
                    f"using {name} = {cpp_method(method)};",
                    f"static_assert( tribol::SupportedMethod<{name}> );",
                ]
            )
            for execution in combination["execution"]:
                lines.append(
                    f"static_assert( tribol::execution::SupportedContactExecution<{name}, "
                    f"tribol::execution::{execution}> );"
                )
            lines.append("")
    return "\n".join(lines)
