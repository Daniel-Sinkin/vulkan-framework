#!/usr/bin/env python3
from __future__ import annotations

import argparse
import site
import sys
from pathlib import Path

try:
    from PIL import Image, ImageStat
except ModuleNotFoundError:
    project_venv = Path(__file__).resolve().parents[1] / ".venv"
    for site_packages in (project_venv / "lib").glob("python*/site-packages"):
        site.addsitedir(str(site_packages))
    from PIL import Image, ImageStat


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate that a ds_vk screenshot is nonblank.")
    parser.add_argument("image")
    parser.add_argument("--min-stddev", type=float, default=3.0)
    parser.add_argument("--min-brightness", type=float, default=8.0)
    parser.add_argument("--max-brightness", type=float, default=247.0)
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_file():
        print(f"[FAIL] missing screenshot: {path}", file=sys.stderr)
        return 1

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        stat = ImageStat.Stat(rgba)
        mean_rgb = sum(stat.mean[:3]) / 3.0
        stddev_rgb = sum(stat.stddev[:3]) / 3.0
        width, height = rgba.size

    ok = (
        stddev_rgb >= args.min_stddev
        and args.min_brightness <= mean_rgb <= args.max_brightness
        and width > 16
        and height > 16
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {path} size={width}x{height} "
        f"mean_rgb={mean_rgb:.2f} stddev_rgb={stddev_rgb:.2f}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
