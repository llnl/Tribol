#!/usr/bin/env python3
"""Plot the EnergyMortar in-bounds cubic projection smoothing map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def cubic_in_bounds_smoothing(xi: np.ndarray, delta: float) -> np.ndarray:
    """Map raw projection coordinate xi into [0, 1] with C1 cubic endpoint smoothing."""
    xi_hat = xi.copy()

    left = (0.0 < xi) & (xi < delta)
    s_left = xi[left] / delta
    xi_hat[left] = delta * (2.0 * s_left * s_left - s_left * s_left * s_left)

    right = (1.0 - delta < xi) & (xi < 1.0)
    s_right = (1.0 - xi[right]) / delta
    xi_hat[right] = 1.0 - delta * (2.0 * s_right * s_right - s_right * s_right * s_right)

    xi_hat = np.where(xi <= 0.0, 0.0, xi_hat)
    xi_hat = np.where(xi >= 1.0, 1.0, xi_hat)
    return xi_hat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("projection_cubic_smoothing.png"))
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Projection smoothing width in raw element coordinates. Must be in (0, 0.5).",
    )
    args = parser.parse_args()
    if args.delta <= 0.0 or args.delta >= 0.5:
        parser.error("--delta must be in (0, 0.5).")

    xi = np.linspace(-0.15, 1.15, 1400)
    xi_hat = cubic_in_bounds_smoothing(xi, args.delta)
    identity = xi

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(xi, xi_hat, color="#0032a1", linewidth=2.75, label=r"$\hat{\xi}(\xi)$")
    ax.plot(xi, identity, color="#6e6e7c", linestyle=":", linewidth=1.75, label="identity")
    ax.axvline(0.0, color="#6e6e7c", linestyle="--", linewidth=1.5)
    ax.axvline(args.delta, color="#ff7900", linestyle="--", linewidth=1.75, label=r"$\delta$")
    ax.axvline(1.0 - args.delta, color="#ff7900", linestyle="--", linewidth=1.75, label=r"$1-\delta$")
    ax.axvline(1.0, color="#6e6e7c", linestyle="--", linewidth=1.5)
    ax.fill_between(
        xi,
        xi_hat,
        identity,
        where=((0.0 <= xi) & (xi <= args.delta)) | ((1.0 - args.delta <= xi) & (xi <= 1.0)),
        color="#3366cc",
        alpha=0.12,
        label="cubic smoothing",
    )

    ax.set_xlabel(r"Projection coordinate $\xi$")
    ax.set_ylabel(r"Smoothed coordinate $\hat{\xi}$")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, args.delta, 0.5, 1.0 - args.delta, 1.0])
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(loc="upper left", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
