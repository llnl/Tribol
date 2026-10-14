# Residual-gap edge-to-cap sweep

Build Tribol with Enzyme and examples enabled, then run the single-process example:

```sh
cmake --build build --target residual_gap_slide_ex
build/examples/residual_gap_slide_ex residual_gap_slide.csv
python3 scripts/plot_residual_gap_slide.py residual_gap_slide.csv --output residual_gap_slide.png
```

The plotting script requires matplotlib. It is optional for building or running Tribol.

The fixed source edge runs from (-1, 0) to (0, 0), with outward normal pointing down. A target edge of length 0.04 moves in the positive x direction at y = -0.1, from center x = -0.8 to 0.4. The residual gap is 0.2, penalty stiffness is 3, and smoothing length is 0.1. The target normal opposes the source normal. Open endpoint caps complete the source offset surface.

The CSV separates projected-edge, endpoint-cap, and summed contact energies and target forces. Forces use Tribol's contact-residual convention (positive energy gradients); a mechanical restoring force uses the opposite sign. The 2-by-2 Jacobian is the derivative of that summed residual with respect to rigid translation of the target edge, including motion of projection bounds, normals, and quadrature. Nodal source and neighbor reactions enter the calculation but are not plotted.

The plotted quantities need not stay constant during the sweep. The regression tests separately check one-sided force and Jacobian limits at handoffs and compare derivatives against finite differences. This example makes the shape and magnitude of the transition visible; visual smoothness alone is not a derivative test. Angular attenuation and full MFEM assembly have separate tests.
