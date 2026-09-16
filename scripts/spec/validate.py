#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spec_model import SpecError, SpecFiles, emit_cpp, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Tribol's normative requirement and capability manifests.")
    parser.add_argument("--requirements", type=Path, default=SpecFiles().requirements)
    parser.add_argument("--capabilities", type=Path, default=SpecFiles().capabilities)
    parser.add_argument("--emit-cpp", action="store_true", help="emit representative C++ method assertions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        requirements, capabilities = validate(SpecFiles(args.requirements, args.capabilities))
    except SpecError as error:
        print(f"spec validation failed: {error}", file=sys.stderr)
        return 1

    if args.emit_cpp:
        print(emit_cpp(capabilities), end="")
    else:
        print(
            f"validated {len(requirements['requirements'])} requirements and "
            f"{len(capabilities['combinations'])} capability combinations"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
