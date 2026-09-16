# Script Instructions

- Keep verification scripts dependency-light and deterministic.
- Provide `--help`, return nonzero on failed required checks, and distinguish unavailable optional tools from failures.
- Resolve repository paths relative to the script location rather than the caller's working directory.
- Add focused tests for parsers, generated output, and architecture checks.
- Do not silently rewrite user files from a check command.
