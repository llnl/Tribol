#!/usr/bin/env python3
"""Plot the EnergyMortar eta angle-smoothing weight."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def eta_weight(theta: np.ndarray, start_angle: float) -> np.ndarray:
    theta0 = start_angle
    theta1 = math.pi / 2.0
    t = (theta - theta0) / (theta1 - theta0)
    smooth = 3.0 * t * t - 2.0 * t * t * t
    weight = np.where(theta <= theta0, 1.0, np.where(theta >= theta1, 0.0, 1.0 - smooth))
    return weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("eta_angle_smoothing_weight.png"))
    parser.add_argument(
        "--start-angle-degrees",
        type=float,
        default=45.0,
        help="Angle in degrees where eta smoothing starts; smoothing ends at 90 degrees.",
    )
    args = parser.parse_args()
    if args.start_angle_degrees < 0.0 or args.start_angle_degrees >= 90.0:
        parser.error("--start-angle-degrees must be in [0, 90).")

    start_angle = math.radians(args.start_angle_degrees)
    end_angle = math.pi / 2.0
    theta = np.linspace(0.0, end_angle, 1000)
    weight = eta_weight(theta, start_angle)

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
    ax.plot(np.degrees(theta), weight, color="#0032a1", linewidth=2.75, label=r"$w_\eta(\theta)$")
    ax.axvline(args.start_angle_degrees, color="#ff7900", linestyle="--", linewidth=1.75, label="taper starts")
    ax.axvline(90.0, color="#6e6e7c", linestyle=":", linewidth=1.75, label="90°")
    ax.fill_between(
        np.degrees(theta),
        weight,
        0.0,
        where=theta >= start_angle,
        color="#3366cc",
        alpha=0.12,
        label="smoothed range",
    )

    ax.set_xlabel(r"Angle $\theta$ (degrees)")
    ax.set_ylabel(r"Weight $w_\eta(\theta)$")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0.0, 30.0, 60.0, args.start_angle_degrees, 90.0])
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(loc="lower left", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
