"""Extract individual circular artwork assets from this reviewed reference.

Coordinates are reviewed, normalized to the full frame. This is not automatic
object segmentation. Never exports a full-frame backdrop.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ICONS = {
    "moneyprinter": (.3105, .153, .0193),
    "forge": (.496, .132, .024),
    "focus": (.669, .155, .0185),
    "appflowy": (.815, .178, .0225),
    "penpot": (.257, .305, .019),
    "headroom": (.850, .317, .0195),
    "localrecall": (.872, .454, .020),
    "markitdown": (.838, .591, .0195),
    "vision": (.109, .593, .020),
    "guard": (.195, .683, .020),
    "reach": (.731, .712, .025),
}


def extract(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGBA")
        w, h = image.size
        for name, (x, y, radius) in ICONS.items():
            r = radius * w
            crop = image.crop((round(x*w-r), round(y*h-r), round(x*w+r), round(y*h+r)))
            alpha = Image.new("L", crop.size)
            ImageDraw.Draw(alpha).ellipse((1, 1, crop.width-2, crop.height-2), fill=255)
            alpha = alpha.filter(ImageFilter.GaussianBlur(.7))
            crop.putalpha(ImageChops.multiply(crop.getchannel("A"), alpha))
            crop.save(destination / f"{name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    extract(args.source, args.destination)
