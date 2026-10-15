# Tribol Contact Examples

The examples use the policy-composed C++ interface from `tribol/Tribol.hpp`. Configure with
`-DENABLE_EXAMPLES=ON` and build an example by target name.

## Mortar patch test

`mortar_patch_test.cpp` places one coarse mortar quadrilateral above four nonmortar quadrilaterals. The two surfaces
cover the same unit square but have nonmatching meshes. A constant Lagrange multiplier applies a uniform normal
traction across their projected overlap.

The executable verifies the defining patch-test properties:

- the search finds all four coarse-to-fine interactions;
- the integrated force equals pressure times area;
- mortar and nonmortar forces are equal and opposite;
- the constant traction produces the expected nodal force distribution on both meshes; and
- the integrated gap constraint equals gap times overlap area.

Build and run it with:

```bash
cmake --build <build-directory> --target tribol_mortar_patch_test
<build-directory>/examples/tribol_mortar_patch_test
```

The executable returns a nonzero status when any patch-test check fails, so test-enabled builds also register it with
CTest as `tribol_mortar_patch_test`.
