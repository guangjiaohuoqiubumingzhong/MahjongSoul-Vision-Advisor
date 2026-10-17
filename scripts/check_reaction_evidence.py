"""Re-check last-discard highlight and reaction banners on saved live frames.

For every saved frame under reports/screenshot-advice with its advice.json,
the river boxes from that run are reused and only the reaction evidence is
recomputed with the current code. Prints one line per frame and a summary.

    .venv\\Scripts\\python scripts\\check_reaction_evidence.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.capture.frames import load_image
from app.vision.actions import call_banners, highlighted_discard
from app.vision.config import load_config, pixel_rect, to_reference, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/window-steam-1366-v2.json"))
    parser.add_argument("--runs", default="reports/screenshot-advice/run-*/**/advice.json")
    parser.add_argument("--output", type=Path, default=Path("reports/vision/reaction-evidence.json"))
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    config = load_config(args.config)
    buttons = config["regions"]["buttons"]
    x, y, w, h = pixel_rect(buttons.get("banner_rect", buttons["rect"]), *config["client_size"])
    rows, summary = [], Counter()
    for name in sorted(glob.glob(args.runs, recursive=True)):
        path = Path(name)
        frames = [p for p in path.parent.glob("[0-9]*.png")]
        advice = json.loads(path.read_text(encoding="utf-8"))
        observation = advice.get("observation")
        if not frames or not observation:
            continue
        native = load_image(frames[0])
        try:
            reference, _, _ = to_reference(native, config)
        except Exception as exc:  # unsupported size
            print(f"{path.parent}: skipped ({exc})")
            continue
        highlight = highlighted_discard(reference, observation["regions"], config["seat_mapping"]["self"])
        banners = [item["label"] for item in call_banners(reference[y:y+h, x:x+w])]
        old = observation.get("reaction_discard") or {}
        row = {"frame": str(frames[0]), "status": advice.get("status"), "reason": advice.get("reason"),
               "old_target": old.get("target") if old.get("confirmed") else None,
               "target": highlight.get("target"), "tile": highlight.get("tile"),
               "glowing_tiles": [(g["region"], g["tile"], g["glow_density"]) for g in highlight["glowing_tiles"]],
               "banners": banners}
        rows.append(row)
        summary["frames"] += 1
        summary["highlight_confirmed"] += highlight["confirmed"]
        summary["highlight_glow_but_unconfirmed"] += bool(highlight["glowing_tiles"]) and not highlight["confirmed"]
        summary["with_banners"] += bool(banners)
        summary["banners_without_highlight"] += bool(banners) and not highlight["confirmed"]
        summary["highlight_without_banners"] += highlight["confirmed"] and not banners
        print(f"{path.parent.parent.name}/{path.parent.name}: {advice.get('status')} "
              f"source={row['target']} {row['tile']} (old {row['old_target']}) banners={banners} "
              f"glow={row['glowing_tiles']}")
    write_json(args.output, {"summary": dict(summary), "frames": rows})
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
