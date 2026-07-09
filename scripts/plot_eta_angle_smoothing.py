#!/usr/bin/env python3
"""Plot the EnergyMortar eta angle-smoothing weight without external packages."""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path


WIDTH = 1800
HEIGHT = 1080
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID = (220, 220, 220)
AXIS = (40, 40, 40)
MARK = (135, 135, 135)

FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "(": ["010", "100", "100", "100", "100", "100", "010"],
    ")": ["010", "001", "001", "001", "001", "001", "010"],
    "/": ["001", "001", "010", "010", "010", "100", "100"],
    ".": ["000", "000", "000", "000", "000", "110", "110"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "E": ["111", "100", "100", "111", "100", "100", "111"],
    "g": ["000", "111", "101", "111", "001", "101", "111"],
    "h": ["100", "100", "100", "111", "101", "101", "101"],
    "i": ["010", "000", "110", "010", "010", "010", "111"],
    "l": ["110", "010", "010", "010", "010", "010", "111"],
    "n": ["000", "000", "110", "101", "101", "101", "101"],
    "p": ["000", "110", "101", "101", "110", "100", "100"],
    "r": ["000", "000", "101", "110", "100", "100", "100"],
    "t": ["010", "010", "111", "010", "010", "010", "001"],
    "w": ["000", "000", "101", "101", "101", "111", "101"],
    "_": ["000", "000", "000", "000", "000", "000", "111"],
}


def eta_weight(theta: float, start_angle: float) -> float:
    theta0 = start_angle
    theta1 = math.pi / 2.0
    if theta <= theta0:
        return 1.0
    if theta >= theta1:
        return 0.0
    t = (theta - theta0) / (theta1 - theta0)
    smooth = 3.0 * t * t - 2.0 * t * t * t
    return 1.0 - smooth


def set_pixel(img: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        i = 3 * (y * WIDTH + x)
        img[i : i + 3] = bytes(color)


def draw_line(
    img: bytearray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    half = max(width // 2, 0)
    while True:
        for yy in range(y0 - half, y0 + half + 1):
            for xx in range(x0 - half, x0 + half + 1):
                set_pixel(img, xx, yy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def text_size(text: str, scale: int) -> tuple[int, int]:
    return len(text) * 4 * scale - scale, 7 * scale


def draw_text(
    img: bytearray, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 4, rotate: bool = False
) -> None:
    if rotate:
        char_h = 7 * scale
        cursor_y = y
        for ch in text:
            draw_text(img, x, cursor_y, ch, color, scale)
            cursor_y += char_h + scale
        return

    cursor = x
    for ch in text:
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, value in enumerate(row):
                if value == "1":
                    for yy in range(scale):
                        for xx in range(scale):
                            set_pixel(img, cursor + gx * scale + xx, y + gy * scale + yy, color)
        cursor += 4 * scale


def write_png(path: Path, img: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = bytearray()
    stride = WIDTH * 3
    for y in range(HEIGHT):
        rows.append(0)
        rows.extend(img[y * stride : (y + 1) * stride])

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(rows), 9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(png)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("eta_angle_smoothing_weight.png"))
    parser.add_argument(
        "--start-angle-degrees",
        type=float,
        default=80.0,
        help="Angle in degrees where eta smoothing starts; smoothing ends at 90 degrees.",
    )
    args = parser.parse_args()
    if args.start_angle_degrees < 0.0 or args.start_angle_degrees >= 90.0:
        parser.error("--start-angle-degrees must be in [0, 90).")
    start_angle = args.start_angle_degrees * math.pi / 180.0

    img = bytearray(WHITE * (WIDTH * HEIGHT))
    left, right = 220, 1680
    top, bottom = 120, 860

    def px(theta: float) -> int:
        return round(left + theta / (math.pi / 2.0) * (right - left))

    def py(value: float) -> int:
        return round(bottom - value * (bottom - top))

    xticks = [
        (0.0, "0"),
        (math.pi / 6.0, "pi/6"),
        (math.pi / 3.0, "pi/3"),
        (start_angle, f"{args.start_angle_degrees:g}"),
        (math.pi / 2.0, "pi/2"),
    ]
    yticks = [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1")]

    for value, label in yticks:
        y = py(value)
        draw_line(img, left, y, right, y, GRID, 2)
        draw_line(img, left - 12, y, left, y, AXIS, 3)
        w, h = text_size(label, 4)
        draw_text(img, left - 28 - w, y - h // 2, label, BLACK, 4)

    for theta, label in xticks:
        x = px(theta)
        draw_line(img, x, top, x, bottom, GRID, 2)
        draw_line(img, x, bottom, x, bottom + 12, AXIS, 3)
        w, _ = text_size(label, 4)
        draw_text(img, x - w // 2, bottom + 28, label, BLACK, 4)

    draw_line(img, left, bottom, right, bottom, AXIS, 4)
    draw_line(img, left, top, left, bottom, AXIS, 4)
    draw_line(img, px(start_angle), top, px(start_angle), bottom, MARK, 3)
    draw_line(img, px(math.pi / 2.0), top, px(math.pi / 2.0), bottom, MARK, 3)

    points = []
    for i in range(900):
        theta = (math.pi / 2.0) * i / 899.0
        points.append((px(theta), py(eta_weight(theta, start_angle))))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        draw_line(img, x0, y0, x1, y1, BLACK, 5)

    xlabel = "Angle theta (rad)"
    w, _ = text_size(xlabel, 5)
    draw_text(img, (left + right) // 2 - w // 2, HEIGHT - 115, xlabel, BLACK, 5)
    draw_text(img, 40, 245, "w_eta(theta)", BLACK, 5, rotate=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.output, img)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
