"""Visible local countdown and seat wind, independent of tile recognition."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import cv2
import numpy as np

from .config import pixel_rect, VisionError


def parse_timer(text):
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    # Only presence matters. Do not constrain digit count, seconds, or require
    # OCR to decode the surrounding decoration / reserve separator exactly.
    match = re.search(r"([0-9]+)(?:\+([0-9]+))?", text)
    if not match:
        return None
    fixed = int(match[1])
    reserve = int(match[2]) if match[2] is not None else None
    return {"fixed_seconds": fixed, "reserve_seconds": reserve}


def detect_timer(pixels, config, reader):
    if not config or reader is None:
        return {"active": False, "reason": "timer_not_configured"}
    x,y,w,h = pixel_rect(config["rect"], pixels.shape[1], pixels.shape[0])
    crop = pixels[y:y+h, x:x+w]
    return detect_timer_crop(crop, config, reader, origin=(x,y))


def detect_timer_crop(crop, config, reader, origin=(0,0)):
    if not config or reader is None:
        return {"active":False, "reason":"timer_not_configured"}
    x,y = origin
    h,w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[:,:,1] < 65) & (hsv[:,:,2] > 190)
    orange = (hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 38) & (hsv[:,:,1] > 110) & (hsv[:,:,2] > 180)
    evidence = {"bbox":[x,y,w,h], "white_fill":float(white.mean()), "orange_fill":float(orange.mean())}
    if (white | orange).mean() < .005:
        return {**evidence, "active":False, "reason":"timer_colours_absent"}
    # The countdown can be a single white/orange number, or fixed + reserve.
    # Its single-number layout starts farther left. Remove small background
    # strokes before OCR rather than requiring both colours or a plus sign.
    count, labels, stats, _ = cv2.connectedComponentsWithStats((white | orange).astype(np.uint8))
    glyphs = [i for i in range(1, count)
              if stats[i, cv2.CC_STAT_HEIGHT] >= max(8, round(h * .3))
              and stats[i, cv2.CC_STAT_AREA] >= max(20, round(w * h * .002))]
    if not glyphs:
        return {**evidence, "active":False, "reason":"timer_digits_absent"}
    yy, xx = np.where(np.isin(labels, glyphs))
    left, top = max(0, int(xx.min())-4), max(0, int(yy.min())-4)
    right, bottom = min(w, int(xx.max())+5), min(h, int(yy.max())+5)
    evidence["text_bbox"] = [x+left, y+top, right-left, bottom-top]
    raw, score = reader._infer(crop[top:bottom, left:right])
    value = parse_timer(raw)
    active = value is not None and score >= config.get("threshold", .95)
    return {**evidence, "active":active, "value":value if active else None,
            "raw_text":raw, "score":float(score), "score_kind":"not_probability",
            "reason":None if active else "timer_ocr_unconfirmed"}


WIND_CLEAR_MIN_SCORE = .85
WIND_CLEAR_MIN_MARGIN = .5


class WindReader:
    def __init__(self, config):
        self.config = config
        self.entries = []
        if not config:
            return
        path = Path(config["manifest"])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest["templates"]:
            file = path.parent / entry["file"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
                raise VisionError("自风模板校验失败。")
            self.entries.append((entry["wind"], cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)))

    def read(self, pixels):
        if not self.config:
            return {"value":None, "reason":"wind_not_configured"}
        x,y,w,h = pixel_rect(self.config["rect"], pixels.shape[1], pixels.shape[0])
        crop = cv2.cvtColor(pixels[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        scores = []
        for wind, template in self.entries:
            resized = cv2.resize(crop, (template.shape[1], template.shape[0]))
            score = float(cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)[0,0])
            scores.append((score, wind))
        scores.sort(reverse=True)
        top, second = scores[0], scores[1]
        # A glow or the turn bar beside the label can pull a clear match just
        # under the threshold (measured 0.958 with the next wind at -0.26);
        # a very large margin over every other wind is then still decisive.
        accepted = ((top[0] >= self.config.get("threshold",.96) and top[0]-second[0] >= .04)
                    or (top[0] >= WIND_CLEAR_MIN_SCORE and top[0]-second[0] >= WIND_CLEAR_MIN_MARGIN))
        return {"value":top[1] if accepted else None, "bbox":[x,y,w,h], "score":top[0],
                "margin":top[0]-second[0], "reason":None if accepted else "wind_unconfirmed"}
