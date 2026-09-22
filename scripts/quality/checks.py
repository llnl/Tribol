#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

NEW_ARCHITECTURE_DIRECTORIES = (
    Path("src/tribol/Tribol.hpp"),
    Path("src/tribol/core"),
    Path("src/tribol/method"),
    Path("src/tribol/contact"),
    Path("src/tribol/evaluation"),
    Path("src/tribol/basis"),
    Path("src/tribol/constraint"),
    Path("src/tribol/integration"),
    Path("src/tribol/geom/ProjectedOverlap.hpp"),
    Path("src/tribol/geom/ConformingOverlap.hpp"),
    Path("src/tribol/search/Search.hpp"),
    Path("src/tribol/execution"),
    Path("src/tribol/assembly"),
    Path("src/tribol/adapters"),
)

DEPENDENCY_FREE_DIRECTORIES = tuple(
    path for path in NEW_ARCHITECTURE_DIRECTORIES if path != Path("src/tribol/adapters")
) + (Path("src/tribol/adapters/array"),)

TEST_DIRECTORIES = (
    Path("src/tests/spec"),
    Path("src/tests/components"),
    Path("src/tests/adapters"),
    Path("src/tests/parallel"),
    Path("src/tests/legacy_parity"),
    Path("src/tests/install"),
)

BENCHMARK_DIRECTORIES = (
    Path("benchmarks/driver"),
    Path("scripts/benchmarks"),
)

SOURCE_SUFFIXES = {".hpp", ".cpp", ".inl", ".cu"}

DEVICE_BACKEND_ADAPTERS = {
    Path("src/tribol/execution/CudaContact.cu"),
    Path("src/tribol/execution/CudaPenalty.cu"),
    Path("src/tribol/execution/DeviceBackendCuda.hpp"),
    Path("src/tribol/execution/DeviceBackendHip.hpp"),
    Path("src/tribol/execution/HipContact.cpp"),
    Path("src/tribol/execution/HipPenalty.cpp"),
}

SHARED_DEVICE_FILES = (
    Path("src/tribol/execution/DeviceContactImplementation.inl"),
    Path("src/tribol/execution/DevicePenaltyImplementation.inl"),
    Path("src/tribol/execution/CudaContactMemory.inl"),
    Path("src/tribol/execution/CudaContactSearch.inl"),
    Path("src/tribol/execution/CudaContactKernels.inl"),
    Path("src/tribol/execution/CudaContactWorkspacePublic.inl"),
    Path("src/tribol/execution/CudaContactWorkspacePrivate.inl"),
)

THIRD_PARTY_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"](?:mfem|mpi|axom|RAJA|umpire|redecomp|cuda|hip)', re.IGNORECASE
)
NAMED_METHOD_DECLARATION = re.compile(
    r"\b(?:class|struct)\s+(?:CommonPlane|SingleMortar|AlignedMortar|EnergyMortar)\b"
)
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
REQUIREMENT_ANNOTATION = re.compile(r"Requirements:\s*[A-Z]+-\d{3}(?:\s*,\s*[A-Z]+-\d{3})*")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else str(self.path)
        return f"{location}: {self.message}"


def files_in_entries(root: Path, entries: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        absolute = root / entry
        if absolute.is_file() and absolute.suffix in SOURCE_SUFFIXES:
            files.append(absolute)
        elif absolute.is_dir():
            files.extend(path for path in absolute.rglob("*") if path.suffix in SOURCE_SUFFIXES)
    return sorted(files)


def architecture_files(root: Path) -> list[Path]:
    return files_in_entries(root, NEW_ARCHITECTURE_DIRECTORIES)


def check_dependency_boundaries(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in files_in_entries(root, DEPENDENCY_FREE_DIRECTORIES):
        if path.suffix == ".cu" or path.relative_to(root) in DEVICE_BACKEND_ADAPTERS:
            continue
        relative = path.relative_to(root)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if THIRD_PARTY_INCLUDE.search(line):
                findings.append(
                    Finding(relative, line_number, "new architecture code may not include third-party APIs")
                )
    return findings


def check_device_backend_isolation(root: Path) -> list[Finding]:
    forbidden = re.compile(
        r"(?:#\s*include\s*[<\"](?:cuda|hip)|\b(?:cuda|hip)(?:Malloc|Free|Memcpy|Memset|Device|Stream|Get)|"
        r"\b(?:cub|hipcub)::|__global__|<<<|>>>)"
    )
    findings: list[Finding] = []
    for relative in SHARED_DEVICE_FILES:
        path = root / relative
        if not path.exists():
            findings.append(Finding(relative, 0, "missing shared device implementation file"))
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if forbidden.search(line):
                findings.append(
                    Finding(relative, line_number, "shared device code contains a backend-specific runtime token")
                )
    return findings


def check_named_method_classes(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in architecture_files(root):
        relative = path.relative_to(root)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = NAMED_METHOD_DECLARATION.search(line)
            if match:
                findings.append(
                    Finding(relative, line_number, f"forbidden named method declaration: {match.group(0)}")
                )
    return findings


def check_file_sizes(root: Path, production_limit: int = 500, test_limit: int = 800) -> list[Finding]:
    findings: list[Finding] = []
    paths = architecture_files(root)
    spec_root = root / "src/tests/spec"
    if spec_root.exists():
        paths.extend(path for path in spec_root.rglob("*") if path.suffix in {".hpp", ".cpp"})
    paths.extend(path for path in files_in_entries(root, BENCHMARK_DIRECTORIES) if path.suffix in SOURCE_SUFFIXES)
    for path in sorted(set(paths)):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        relative = path.relative_to(root)
        limit = test_limit if "tests" in relative.parts else production_limit
        if lines > limit:
            findings.append(Finding(relative, 0, f"{lines} lines exceeds the {limit}-line limit"))
    return findings


def _listed_core_headers(cmake_text: str) -> set[str]:
    match = re.search(r"set\(tribol_core_headers\s+(.*?)\)", cmake_text, re.DOTALL)
    if not match:
        return set()
    return {token for token in re.split(r"\s+", match.group(1).strip()) if token}


def check_public_header_manifest(root: Path) -> list[Finding]:
    cmake_path = root / "src/tribol/core/CMakeLists.txt"
    if not cmake_path.exists():
        return [Finding(cmake_path.relative_to(root), 0, "missing core header manifest")]
    listed = _listed_core_headers(cmake_path.read_text(encoding="utf-8"))
    adapter_cmake_path = root / "src/tribol/adapters/mfem/CMakeLists.txt"
    adapter_text = adapter_cmake_path.read_text(encoding="utf-8") if adapter_cmake_path.exists() else ""
    findings: list[Finding] = []
    for path in architecture_files(root):
        if path.suffix != ".hpp" or path.name == "AGENTS.md" or path.relative_to(root) in DEVICE_BACKEND_ADAPTERS:
            continue
        relative_to_core = Path("../") / path.relative_to(root / "src/tribol")
        expected = path.name if path.parent == root / "src/tribol/core" else relative_to_core.as_posix()
        in_adapter_manifest = path.parent == root / "src/tribol/adapters/mfem" and path.name in adapter_text
        if expected not in listed and not in_adapter_manifest:
            findings.append(
                Finding(path.relative_to(root), 0, f"public header is absent from {cmake_path.relative_to(root)}")
            )
    return findings


def check_install_contract(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    contracts = {
        Path("cmake/Options.cmake"): (
            'option(TRIBOL_ENABLE_BENCHMARKS "Build rewritten Tribol benchmark drivers" OFF)',
        ),
        Path("src/tribol/core/CMakeLists.txt"): ("EXPORT_NAME tribol::core",),
        Path("src/tribol/adapters/mfem/CMakeLists.txt"): ("EXPORT_NAME tribol::mfem_adapter",),
        Path("src/tests/install/CMakeLists.txt"): (
            "find_package(tribol CONFIG REQUIRED)",
            "tribol::core",
            "tribol::mfem_adapter",
        ),
        Path("src/tests/CMakeLists.txt"): ("NAME tribol_install_consumer",),
    }
    for relative, required_text in contracts.items():
        path = root / relative
        if not path.exists():
            findings.append(Finding(relative, 0, "missing install-consumer contract file"))
            continue
        text = path.read_text(encoding="utf-8")
        for expected in required_text:
            if expected not in text:
                findings.append(Finding(relative, 0, f"missing install-consumer contract: {expected}"))
    return findings


def check_benchmark_contract(root: Path) -> list[Finding]:
    required = (
        Path("benchmarks/suites.json"),
        Path("benchmarks/driver/BenchmarkProtocol.hpp"),
        Path("benchmarks/driver/RewrittenDriver.cpp"),
        Path("benchmarks/driver/LegacyDriver.cpp"),
        Path("benchmarks/reference/CMakeLists.txt"),
        Path("scripts/benchmarks/compare.py"),
    )
    findings = [
        Finding(path, 0, "missing cross-version benchmark contract file")
        for path in required
        if not (root / path).is_file()
    ]
    manifest_path = root / "benchmarks/suites.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            findings.append(Finding(manifest_path.relative_to(root), 0, f"invalid benchmark manifest: {error}"))
        else:
            required_suites = {"smoke", "physics", "scaling", "all"}
            missing = required_suites - set(manifest.get("suites", {}))
            for suite in sorted(missing):
                findings.append(Finding(manifest_path.relative_to(root), 0, f"missing benchmark suite: {suite}"))
    compare_path = root / "scripts/benchmarks/compare.py"
    if compare_path.is_file():
        text = compare_path.read_text(encoding="utf-8")
        for option in ("--reference-dir", "--reference-url", "--reference-ref"):
            if option not in text:
                findings.append(Finding(compare_path.relative_to(root), 0, f"missing reference selector: {option}"))
    return findings


def check_legacy_sources_absent(root: Path) -> list[Finding]:
    prohibited = (
        Path("data"),
        Path("src/examples"),
        Path("src/redecomp"),
        Path("src/shared"),
        Path("src/tribol/common"),
        Path("src/tribol/future"),
        Path("src/tribol/integ"),
        Path("src/tribol/interface"),
        Path("src/tribol/mesh"),
        Path("src/tribol/physics"),
        Path("src/tribol/utils"),
        Path("src/tribol/geom/CompGeom.hpp"),
        Path("src/tribol/search/InterfacePairFinder.hpp"),
    )
    findings = []
    for path in prohibited:
        absolute = root / path
        contains_files = absolute.is_file() or (
            absolute.is_dir() and any(entry.is_file() for entry in absolute.rglob("*"))
        )
        if contains_files:
            findings.append(Finding(path, 0, "legacy implementation path must remain absent"))
    findings.extend(
        Finding(path.relative_to(root), 0, "legacy root-level test must remain absent")
        for path in sorted((root / "src/tests").glob("tribol_*.cpp"))
    )
    return findings


def check_requirement_annotations(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    spec_root = root / "src/tests/spec"
    if not spec_root.exists():
        return [Finding(spec_root.relative_to(root), 0, "missing normative specification tests")]
    test_paths = [path for path in files_in_entries(root, TEST_DIRECTORIES) if path.suffix in {".cpp", ".cu"}]
    for path in sorted(test_paths):
        text = path.read_text(encoding="utf-8")
        if not REQUIREMENT_ANNOTATION.search(text):
            findings.append(Finding(path.relative_to(root), 0, "missing '// Requirements: ID-000, ...' annotation"))
    return findings


def check_requirement_evidence(root: Path) -> list[Finding]:
    requirements_path = root / "src/tests/spec/requirements.yaml"
    if not requirements_path.exists():
        return [Finding(requirements_path.relative_to(root), 0, "missing requirements manifest")]
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    required_ids = {entry["id"] for entry in requirements.get("requirements", [])}
    evidence_paths = files_in_entries(root, TEST_DIRECTORIES)
    workflow = root / ".github/workflows/quality.yml"
    if workflow.exists():
        evidence_paths.append(workflow)
    evidenced_ids: set[str] = set()
    for path in evidence_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Requirements:" in line:
                evidenced_ids.update(re.findall(r"[A-Z]+-\d{3}", line.split("Requirements:", maxsplit=1)[1]))
    findings = [
        Finding(requirements_path.relative_to(root), 0, f"requirement has no test evidence annotation: {identifier}")
        for identifier in sorted(required_ids - evidenced_ids)
    ]
    findings.extend(
        Finding(path.relative_to(root), 0, f"test evidence references unknown requirement: {identifier}")
        for path in [requirements_path]
        for identifier in sorted(evidenced_ids - required_ids)
    )
    return findings


def check_markdown_links(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    paths = [root / "AGENTS.md"]
    design = root / "docs/design"
    if design.exists():
        paths.extend(sorted(design.rglob("*.md")))
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for target in MARKDOWN_LINK.findall(line):
                if target.startswith(("http://", "https://", "#")):
                    continue
                target_path = target.split("#", maxsplit=1)[0]
                if target_path and not (path.parent / target_path).resolve().exists():
                    findings.append(Finding(path.relative_to(root), line_number, f"broken relative link: {target}"))
    return findings


def run_custom_checks(root: Path) -> list[Finding]:
    checks = (
        check_dependency_boundaries,
        check_device_backend_isolation,
        check_named_method_classes,
        check_file_sizes,
        check_public_header_manifest,
        check_install_contract,
        check_benchmark_contract,
        check_legacy_sources_absent,
        check_requirement_annotations,
        check_requirement_evidence,
        check_markdown_links,
    )
    return [finding for check in checks for finding in check(root)]
