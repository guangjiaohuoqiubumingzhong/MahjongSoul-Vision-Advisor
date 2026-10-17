"""Create labelled, enlarged crops for manual review of saved game frames.

This tool proposes no ground truth. Labels must be transcribed from the image.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from app.capture.frames import load_image, save_image
from app.vision.config import load_config, pixel_rect
from app.vision.detector import tile_boxes
from app.vision.templates import rotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regions", nargs="+")
    args = parser.parse_args()
    config, pixels = load_config(args.config), load_image(args.image)
    names = args.regions or [name for name, region in config["regions"].items()
                             if region["kind"] in {"river", "meld"}]
    panels = []
    for name in names:
        region = config["regions"][name]
        x, y, w, h = pixel_rect(region["rect"], pixels.shape[1], pixels.shape[0])
        crop = rotate(pixels[y:y+h, x:x+w], region.get("rotation", 0))
        shown = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        for i, (bx, by, bw, bh) in enumerate(tile_boxes(crop, region)):
            cv2.rectangle(shown, (bx*3, by*3), ((bx+bw)*3, (by+bh)*3), (0,180,255), 2)
            cv2.putText(shown, str(i), (bx*3, max(20,by*3)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,180,255), 2)
        panel = np.full((shown.shape[0]+30, max(1000, shown.shape[1]),3), 235, np.uint8)
        panel[30:, :shown.shape[1]] = shown
        cv2.putText(panel, f"{name}  ROI {x},{y},{w},{h}  CCW {region.get('rotation',0)}", (5,22),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (20,20,20), 1)
        panels.append(panel)
    width = max(panel.shape[1] for panel in panels)
    panels = [cv2.copyMakeBorder(panel,0,0,0,width-panel.shape[1],cv2.BORDER_CONSTANT,value=(235,235,235))
              for panel in panels]
    save_image(args.out, np.vstack(panels))


if __name__ == "__main__":
    main()
