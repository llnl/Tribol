#!/usr/bin/env python3
# Copyright (c) Lawrence Livermore National Security, LLC and other Tribol Project Developers.
# SPDX-License-Identifier: (MIT)
"""Plot the CSV from residual_gap_slide_ex; requires matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("residual_gap_slide.png"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    x = [float(row["position"]) for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for axis, quantity, label in zip(axes[:3], ("energy", "fx", "fy"), ("Contact energy", "Tangential force", "Normal force")):
        for contribution in ("edge", "cap", "total"):
            axis.plot(x, [float(row[f"{contribution}_{quantity}"]) for row in rows], label=contribution)
        axis.set_ylabel(label)
        axis.legend(loc="best")
    for component in ("kxx", "kxy", "kyy"):
        axes[3].plot(x, [float(row[component]) for row in rows], label=component)
    axes[3].set_ylabel("Translation Jacobian")
    axes[3].legend(loc="best")
    axes[3].set_xlabel("Target-edge center relative to source endpoint")
    for axis in axes:
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.grid(alpha=0.2)
    fig.suptitle("One edge pair: residual gap 0.2, normal separation 0.1")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
