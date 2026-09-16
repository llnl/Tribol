# Normative Tribol Specification

This directory defines the supported public product.

- `requirements.yaml` assigns stable IDs to externally meaningful behavior.
- `capabilities.yaml` lists supported policy families and required environments.
- `SupportedMethods.hpp` expresses representative tuples in C++.
- `tribol_interface_spec.cpp` makes public syntax, rejection rules, traits, views, and lifecycle executable.

The `.yaml` files intentionally use JSON syntax, which is valid YAML 1.2 and can be parsed with Python's standard
library. Run:

```bash
python3 scripts/spec/validate.py
python3 scripts/spec/validate.py --emit-cpp > /tmp/tribol-supported-methods.cpp
clang++ -std=c++20 -Isrc -fsyntax-only /tmp/tribol-supported-methods.cpp
python3 -m unittest discover -s scripts/spec/tests
```

A capability is not complete merely because its tuple compiles. Every referenced requirement needs component,
conformance, adapter, quality, or parallel evidence appropriate to its scope.

Vocabulary lists are closed: an unrecognized topology, search, execution mode, parallel mode, or output is rejected.
Variant entries are expanded into distinct C++ method assertions. Duplicate policy tuples are rejected even when their
human-readable IDs differ.
