#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmark_model import compare_results, expand_suite, human_report, load_manifest, parse_result, report_data

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/suites.json"
REFERENCE_URL = "https://github.com/LLNL/Tribol.git"
DEVICE_CASES = frozenset(("penalty-2d", "penalty-3d"))
CUDA_REFERENCE_SOURCE_PATCHES = (
    (
        "cuda-13-device-array-destructors",
        Path("src/tribol/common/Containers.hpp"),
        (
            (
                "  TRIBOL_DEFAULT_HOST_DEVICE ~DeviceArray() = default;",
                "  TRIBOL_HOST_DEVICE ~DeviceArray() {}",
            ),
            (
                "  TRIBOL_DEFAULT_HOST_DEVICE ~DeviceArray2D() = default;",
                "  TRIBOL_HOST_DEVICE ~DeviceArray2D() {}",
            ),
        ),
    ),
    (
        "cuda-13-contact-plane-destructors",
        Path("src/tribol/geom/CompGeom.hpp"),
        (
            (
                "  virtual ~CompGeomPair() = default;",
                "  TRIBOL_HOST_DEVICE virtual ~CompGeomPair() {}",
            ),
            (
                "  TRIBOL_HOST_DEVICE inline ContactPlanePair(){};\n\n  /**",
                "  TRIBOL_HOST_DEVICE inline ContactPlanePair(){};\n\n"
                "  TRIBOL_HOST_DEVICE ~ContactPlanePair() override {}\n\n  /**",
            ),
            (
                "  ~CommonPlanePair() = default;",
                "  TRIBOL_HOST_DEVICE ~CommonPlanePair() override {}",
            ),
            (
                "  ~MortarPlanePair() = default;",
                "  TRIBOL_HOST_DEVICE ~MortarPlanePair() override {}",
            ),
            (
                "  ~AlignedMortarPlanePair() = default;",
                "  TRIBOL_HOST_DEVICE ~AlignedMortarPlanePair() override {}",
            ),
        ),
    ),
)


def run(command: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    print("+", " ".join(command), file=sys.stderr)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as error:
        if capture:
            if error.stdout:
                print(error.stdout, end="", file=sys.stderr)
            if error.stderr:
                print(error.stderr, end="", file=sys.stderr)
        raise
    if capture:
        return completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    return ""


def cache_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("//", "#")) or ":" not in line or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        values[key] = value
    return values


def git_state(directory: Path) -> dict[str, Any] | None:
    revision = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if revision.returncode != 0:
        return None
    status = subprocess.run(
        ["git", "-C", str(directory), "status", "--short"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return {
        "revision": revision.stdout.strip(),
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def reference_source(args: argparse.Namespace, work: Path) -> Path | None:
    if not args.reference_dir:
        return work / "reference-source"
    reference = args.reference_dir.resolve()
    cache = cache_values(reference / "CMakeCache.txt")
    source = cache.get("CMAKE_HOME_DIRECTORY")
    return Path(source).resolve() if source else None


def apply_cuda_reference_source_compatibility(source: Path) -> list[str]:
    applied: list[str] = []
    for name, relative_path, replacements in CUDA_REFERENCE_SOURCE_PATCHES:
        path = source / relative_path
        if not path.is_file():
            raise ValueError(f"CUDA reference compatibility source is missing: {path}")
        original = path.read_text(encoding="utf-8")
        updated = original
        for old, new in replacements:
            old_count = updated.count(old)
            new_count = updated.count(new)
            if old_count == 1 and new_count == 0:
                updated = updated.replace(old, new)
            elif old_count == 0 and new_count == 1:
                continue
            else:
                raise ValueError(
                    f"CUDA reference compatibility patch {name!r} does not match {path}; "
                    f"expected exactly one original or patched occurrence"
                )
        if updated != original:
            path.write_text(updated, encoding="utf-8")
        applied.append(name)
    return applied


def provenance(
    args: argparse.Namespace,
    work: Path,
    current_driver: Path,
    reference_config: Path,
    reference_driver: Path,
) -> dict[str, Any]:
    source = reference_source(args, work)
    reference: dict[str, Any] = {
        "mode": "directory" if args.reference_dir else "develop",
        "driver": str(reference_driver.resolve()),
        "package_config": str(reference_config.resolve()),
    }
    if args.reference_dir:
        reference["requested_directory"] = str(args.reference_dir.resolve())
    else:
        reference["url"] = args.reference_url
        reference["ref"] = args.reference_ref
    if source:
        reference["source_directory"] = str(source)
        state = git_state(source)
        if state:
            reference["git"] = state
    compatibility = getattr(args, "reference_source_compatibility", [])
    if compatibility:
        reference["source_compatibility"] = compatibility
    return {
        "suite": args.suite,
        "execution": args.execution,
        "manifest": str(args.manifest.resolve()),
        "current": {
            "driver": str(current_driver.resolve()),
            "build_directory": str(args.current_build.resolve()),
            "git": git_state(REPOSITORY_ROOT),
        },
        "reference": reference,
    }


def configure_current(args: argparse.Namespace) -> Path:
    if args.current_driver:
        driver = args.current_driver.resolve()
        if not driver.is_file():
            raise ValueError(f"current driver does not exist: {driver}")
        return driver
    build = args.current_build.resolve()
    command = ["cmake"]
    if args.current_host_config and not (build / "CMakeCache.txt").exists():
        command.extend(("-C", str(args.current_host_config.resolve())))
    command.extend(("-S", str(REPOSITORY_ROOT), "-B", str(build), "-DTRIBOL_ENABLE_BENCHMARKS=ON"))
    if args.execution == "cuda":
        command.extend(("-DENABLE_CUDA=ON", "-DENABLE_HIP=OFF"))
    elif args.execution == "hip":
        command.extend(("-DENABLE_CUDA=OFF", "-DENABLE_HIP=ON"))
    command.extend(args.current_cmake_arg)
    run(command)
    target = {
        "host": "tribol_rewritten_benchmark",
        "cuda": "tribol_rewritten_cuda_benchmark",
        "hip": "tribol_rewritten_hip_benchmark",
    }[args.execution]
    run(["cmake", "--build", str(build), "--target", target, "-j", str(args.jobs)])
    candidates = (build / f"benchmarks/{target}", build / f"bin/{target}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"could not find tribol_rewritten_benchmark under {build}")


def find_package_config(directory: Path) -> Path | None:
    direct = directory / "tribol-config.cmake"
    if direct.is_file():
        return direct
    patterns = ("lib/cmake/tribol/tribol-config.cmake", "lib64/cmake/tribol/tribol-config.cmake")
    for pattern in patterns:
        candidate = directory / pattern
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob("**/tribol-config.cmake")) if directory.is_dir() else []
    return matches[0] if matches else None


def install_reference_build(build: Path, prefix: Path, jobs: int) -> Path:
    run(["cmake", "--build", str(build), "-j", str(jobs)])
    run(["cmake", "--install", str(build), "--prefix", str(prefix)])
    config = find_package_config(prefix)
    if not config:
        raise ValueError(f"installing reference build {build} did not produce a Tribol package")
    return config


def inherited_reference_arguments(current_build: Path, execution: str = "host") -> list[str]:
    cache = cache_values(current_build / "CMakeCache.txt")
    keys = [
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "MPI_C_COMPILER",
        "MPI_C_COMPILER_INCLUDE_DIRS",
        "MPI_C_HEADER_DIR",
        "MPI_CXX_COMPILER",
        "MPI_CXX_COMPILER_INCLUDE_DIRS",
        "MPI_CXX_HEADER_DIR",
        "AXOM_DIR",
        "MFEM_DIR",
        "ENABLE_MPI",
    ]
    if execution == "cuda":
        keys.extend(
            (
                "CMAKE_CUDA_COMPILER",
                "CMAKE_CUDA_HOST_COMPILER",
                "CMAKE_CUDA_ARCHITECTURES",
                "CUDAToolkit_ROOT",
                "CUDA_TOOLKIT_ROOT_DIR",
                "RAJA_DIR",
                "umpire_DIR",
                "UMPIRE_DIR",
            )
        )
    elif execution == "hip":
        keys.extend(
            (
                "CMAKE_HIP_ARCHITECTURES",
                "ROCM_PATH",
                "ROCM_ROOT_DIR",
                "HIP_PATH",
                "HIP_ROOT_DIR",
                "RAJA_DIR",
                "hipcub_DIR",
                "umpire_DIR",
                "UMPIRE_DIR",
            )
        )
    arguments = [f"-D{key}={cache[key]}" for key in keys if cache.get(key) and not cache[key].endswith("-NOTFOUND")]
    if execution == "cuda":
        cuda_flags = " ".join(part.strip() for part in shlex.split(cache.get("CMAKE_CUDA_FLAGS", "")) if part.strip())
        required_flags = (
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            f"--pre-include={REPOSITORY_ROOT / 'benchmarks/reference/LegacyCudaCompatibility.hpp'}",
        )
        for required_flag in required_flags:
            if required_flag not in cuda_flags:
                cuda_flags = f"{cuda_flags} {required_flag}".strip()
        arguments.append(f"-DCMAKE_CUDA_FLAGS={cuda_flags}")
    return arguments


def inherited_reference_driver_arguments(current_build: Path, execution: str = "host") -> list[str]:
    cache = cache_values(current_build / "CMakeCache.txt")
    keys = [
        "CMAKE_CXX_COMPILER",
        "MPI_CXX_COMPILER",
        "MPI_CXX_COMPILER_INCLUDE_DIRS",
        "MPI_CXX_HEADER_DIR",
    ]
    if execution == "cuda":
        keys.extend(
            (
                "CMAKE_CUDA_COMPILER",
                "CMAKE_CUDA_HOST_COMPILER",
                "CMAKE_CUDA_ARCHITECTURES",
                "CUDAToolkit_ROOT",
                "CUDA_TOOLKIT_ROOT_DIR",
            )
        )
    elif execution == "hip":
        keys.extend(
            (
                "CMAKE_HIP_ARCHITECTURES",
                "ROCM_PATH",
                "ROCM_ROOT_DIR",
                "HIP_PATH",
                "HIP_ROOT_DIR",
            )
        )
    return [f"-D{key}={cache[key]}" for key in keys if cache.get(key) and not cache[key].endswith("-NOTFOUND")]


def build_develop_reference(args: argparse.Namespace, work: Path) -> Path:
    source = work / "reference-source"
    build = work / "reference-build"
    prefix = work / "reference-install"
    if not source.exists():
        run(
            [
                "git",
                "clone",
                "--recurse-submodules=cmake/blt",
                "--branch",
                args.reference_ref,
                "--depth",
                "1",
                args.reference_url,
                str(source),
            ]
        )
    if args.execution == "cuda":
        args.reference_source_compatibility = apply_cuda_reference_source_compatibility(source)
    command = ["cmake"]
    if args.reference_host_config and not (build / "CMakeCache.txt").exists():
        command.extend(("-C", str(args.reference_host_config.resolve())))
    command.extend(
        (
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_EXAMPLES=OFF",
            "-DENABLE_DOCS=OFF",
            "-UENZYME_DIR",
            "-UCALIPER_DIR",
        )
    )
    if args.execution == "cuda":
        command.append("-DENABLE_CUDA=ON")
        command.append("-DENABLE_HIP=OFF")
    elif args.execution == "hip":
        command.append("-DENABLE_CUDA=OFF")
        command.append("-DENABLE_HIP=ON")
    else:
        command.extend(("-URAJA_DIR", "-UUMPIRE_DIR"))
    if not args.reference_host_config:
        command.extend(inherited_reference_arguments(args.current_build.resolve(), args.execution))
    command.extend(args.reference_cmake_arg)
    run(command)
    return install_reference_build(build, prefix, args.jobs)


def resolve_reference_config(args: argparse.Namespace, work: Path) -> Path:
    if not args.reference_dir:
        return build_develop_reference(args, work)
    reference = args.reference_dir.resolve()
    if not reference.exists():
        raise ValueError(f"reference directory does not exist: {reference}")
    if (reference / "CMakeCache.txt").is_file():
        return install_reference_build(reference, work / "reference-install", args.jobs)
    config = find_package_config(reference)
    if config:
        return config
    raise ValueError(f"{reference} is neither a Tribol build nor install directory")


def build_reference_driver(config: Path, work: Path, args: argparse.Namespace) -> Path:
    build = work / "reference-driver"
    command = [
        "cmake",
        "-S",
        str(REPOSITORY_ROOT / "benchmarks/reference"),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-Dtribol_DIR={config.parent}",
        f"-DTRIBOL_BENCHMARK_EXECUTION={args.execution}",
    ]
    selected_cache = args.current_build.resolve() / "CMakeCache.txt"
    if args.reference_dir and (args.reference_dir.resolve() / "CMakeCache.txt").is_file():
        selected_cache = args.reference_dir.resolve() / "CMakeCache.txt"
    command.extend(inherited_reference_driver_arguments(selected_cache.parent, args.execution))
    command.extend(args.reference_driver_cmake_arg)
    run(command)
    run(["cmake", "--build", str(build), "--target", "tribol_reference_benchmark", "-j", str(args.jobs)])
    driver = build / "tribol_reference_benchmark"
    if not driver.is_file():
        raise ValueError(f"reference driver build did not produce {driver}")
    return driver


def execute(driver: Path, spec: object) -> dict[str, object]:
    output = run(
        [
            str(driver),
            "--case",
            spec.case,
            "--size",
            str(spec.size),
            "--warmups",
            str(spec.warmups),
            "--iterations",
            str(spec.iterations),
        ],
        capture=True,
    )
    return parse_result(output, spec)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rewritten Tribol with an installed or develop Tribol build.")
    parser.add_argument("--suite", default="smoke", help="suite name from benchmarks/suites.json (default: smoke)")
    parser.add_argument(
        "--execution",
        choices=("host", "cuda", "hip"),
        default="host",
        help="execution backend for both implementations (default: host)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--current-driver", type=Path, help="use an already-built rewritten benchmark executable")
    parser.add_argument("--current-build", type=Path, default=REPOSITORY_ROOT / "build-benchmarks")
    parser.add_argument("--current-host-config", type=Path)
    parser.add_argument("--current-cmake-arg", action="append", default=[], metavar="ARG")
    parser.add_argument("--reference-dir", type=Path, help="Tribol build or install directory; defaults to develop")
    parser.add_argument("--reference-host-config", type=Path)
    parser.add_argument("--reference-cmake-arg", action="append", default=[], metavar="ARG")
    parser.add_argument("--reference-driver-cmake-arg", action="append", default=[], metavar="ARG")
    parser.add_argument("--reference-url", default=REFERENCE_URL)
    parser.add_argument("--reference-ref", default="develop")
    parser.add_argument("--work-dir", type=Path, help="persistent workspace for reference source and builds")
    parser.add_argument("--output", type=Path, help="write the complete machine-readable JSON report")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    manifest = load_manifest(args.manifest.resolve())
    specs = expand_suite(manifest, args.suite)
    if args.execution in ("cuda", "hip"):
        unsupported = sorted({spec.case for spec in specs} - DEVICE_CASES)
        if unsupported:
            raise ValueError(f"{args.execution.upper()} comparison does not support cases: " + ", ".join(unsupported))
    current_driver = configure_current(args)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="tribol-benchmarks-")
        work = Path(temporary.name)
    try:
        config = resolve_reference_config(args, work)
        reference_driver = build_reference_driver(config, work, args)
        comparisons = []
        for spec in specs:
            current = execute(current_driver, spec)
            reference = execute(reference_driver, spec)
            comparisons.append(compare_results(spec, current, reference))
        report = report_data(
            comparisons,
            provenance(args, work, current_driver, config, reference_driver),
        )
        print(human_report(comparisons))
        if args.output:
            args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
            args.output.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0 if report["exact"] else 2
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"benchmark comparison failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
