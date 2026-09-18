#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from checks import (
    BENCHMARK_DIRECTORIES,
    DEPENDENCY_FREE_DIRECTORIES,
    TEST_DIRECTORIES,
    architecture_files,
    files_in_entries,
    run_custom_checks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Tool:
    name: str
    executables: tuple[str, ...]
    command: tuple[str, ...]


OPTIONAL_TOOLS = (
    Tool(
        "clang-tidy",
        ("clang-tidy-19", "clang-tidy"),
        (
            "--checks=clang-analyzer-*,bugprone-*,performance-*,portability-*",
            "--warnings-as-errors=clang-analyzer-*,bugprone-*,performance-*,portability-*",
            "src/tests/spec/tribol_interface_spec.cpp",
            "--",
            "-std=c++20",
            "-Isrc",
        ),
    ),
    Tool(
        "include-what-you-use",
        ("include-what-you-use", "iwyu"),
        ("-std=c++20", "-Isrc", "src/tests/spec/tribol_interface_spec.cpp"),
    ),
    Tool("cppcheck", ("cppcheck",), ("--enable=warning,style,performance,portability", "--std=c++20")),
    Tool("codespell", ("codespell",), ()),
    Tool("cmake-format", ("cmake-format",), ("--check",)),
    Tool("cmake-lint", ("cmake-lint",), ()),
    Tool("shellcheck", ("shellcheck",), ("scripts/github-actions/linux-check.sh",)),
    Tool("shfmt", ("shfmt",), ("-d", "scripts/github-actions/linux-check.sh")),
    Tool("ruff", ("ruff",), ("check", "scripts/benchmarks", "scripts/quality", "scripts/spec")),
    Tool("mypy", ("mypy",), ("scripts/benchmarks", "scripts/quality", "scripts/spec")),
)

CMAKE_QUALITY_FILES = (
    "benchmarks/CMakeLists.txt",
    "benchmarks/reference/CMakeLists.txt",
    "cmake/TribolQuality.cmake",
    "src/tribol/core/CMakeLists.txt",
    "src/tribol/adapters/mfem/CMakeLists.txt",
    "src/tests/install/CMakeLists.txt",
)


def find_executable(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    return None


def run(command: Sequence[str], root: Path) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=root, check=False).returncode


def custom(root: Path) -> int:
    findings = run_custom_checks(root)
    if findings:
        for finding in findings:
            print(f"ERROR {finding}")
        return 1
    print("PASS custom architecture, manifest, requirement, size, and documentation checks")
    return run((sys.executable, "scripts/spec/validate.py"), root)


def spec_check(root: Path, strict: bool, compiler: str | None) -> int:
    executable = compiler or find_executable(("clang++-19", "clang++", "g++"))
    if not executable:
        print("ERROR C++ compiler is unavailable" if strict else "SKIP C++ compiler is unavailable")
        return 1 if strict else 0
    with tempfile.TemporaryDirectory() as directory:
        translation_unit = Path(directory) / "supported_methods.cpp"
        print(f"+ {sys.executable} scripts/spec/validate.py --emit-cpp > {translation_unit}", flush=True)
        with translation_unit.open("w", encoding="utf-8") as output:
            generated = subprocess.run(
                (sys.executable, "scripts/spec/validate.py", "--emit-cpp"),
                cwd=root,
                check=False,
                stdout=output,
            )
        if generated.returncode:
            return generated.returncode
        return run((executable, "-std=c++20", "-Isrc", "-fsyntax-only", str(translation_unit)), root)


def format_check(root: Path, strict: bool) -> int:
    executable = find_executable(("clang-format-19", "clang-format-18", "clang-format"))
    if not executable:
        print("ERROR clang-format is unavailable" if strict else "SKIP clang-format is unavailable")
        return 1 if strict else 0
    files = {str(path.relative_to(root)) for path in architecture_files(root)}
    files.update(str(path.relative_to(root)) for path in files_in_entries(root, TEST_DIRECTORIES))
    files.update(str(path.relative_to(root)) for path in files_in_entries(root, BENCHMARK_DIRECTORIES))
    return run((executable, "--dry-run", "--Werror", *files), root)


def header_check(root: Path, strict: bool, compiler: str | None) -> int:
    executable = compiler or find_executable(("clang++-19", "clang++", "g++"))
    if not executable:
        print("ERROR C++ compiler is unavailable" if strict else "SKIP C++ compiler is unavailable")
        return 1 if strict else 0
    headers = [path for path in files_in_entries(root, DEPENDENCY_FREE_DIRECTORIES) if path.suffix == ".hpp"]
    with tempfile.TemporaryDirectory() as directory:
        translation_unit = Path(directory) / "header.cpp"
        for header in headers:
            include = header.relative_to(root / "src").as_posix()
            translation_unit.write_text(f'#include "{include}"\nint main() {{ return 0; }}\n', encoding="utf-8")
            result = run((executable, "-std=c++20", "-Isrc", "-fsyntax-only", str(translation_unit)), root)
            if result:
                print(f"ERROR public header is not self-contained: {include}")
                return result
    print(f"PASS {len(headers)} dependency-free public headers compile in isolation")
    return 0


def optional_tools(root: Path, strict: bool) -> int:
    failed = False
    architecture = tuple(str(path.relative_to(root)) for path in architecture_files(root) if path.suffix != ".cu")
    tests = tuple(
        str(path.relative_to(root))
        for path in files_in_entries(root, TEST_DIRECTORIES)
        if path.suffix in {".cpp", ".cu", ".hpp"}
    )
    for tool in OPTIONAL_TOOLS:
        executable = find_executable(tool.executables)
        if not executable:
            status = "ERROR" if strict else "SKIP"
            print(f"{status} {tool.name} is unavailable")
            failed = failed or strict
            continue
        command = tool.command
        if tool.name == "cppcheck":
            command += architecture
        elif tool.name == "codespell":
            command += architecture + tests + (
                "benchmarks",
                "docs/design",
                "scripts/benchmarks",
                "scripts/quality",
                "scripts/spec",
            )
        elif tool.name in {"cmake-format", "cmake-lint"}:
            command += CMAKE_QUALITY_FILES
        failed = run((executable, *command), root) != 0 or failed
    return int(failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tribol rewrite quality checks.")
    parser.add_argument(
        "checks", nargs="*", choices=("custom", "spec", "format", "headers", "tools", "all"), default=["all"]
    )
    parser.add_argument(
        "--strict-tools", action="store_true", help="fail when an optional external tool is unavailable"
    )
    parser.add_argument("--compiler", help="C++ compiler for header-isolation checks")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {"custom", "spec", "format", "headers", "tools"} if "all" in args.checks else set(args.checks)
    operations = {
        "custom": lambda: custom(args.root),
        "spec": lambda: spec_check(args.root, args.strict_tools, args.compiler),
        "format": lambda: format_check(args.root, args.strict_tools),
        "headers": lambda: header_check(args.root, args.strict_tools, args.compiler),
        "tools": lambda: optional_tools(args.root, args.strict_tools),
    }
    failed = False
    for name in ("custom", "spec", "format", "headers", "tools"):
        if name in selected:
            failed = operations[name]() != 0 or failed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
