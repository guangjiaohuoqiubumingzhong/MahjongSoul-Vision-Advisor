"""Print the controlled perspective benchmark as a comparison table."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(path="reports/vision/tile-perspective-benchmark.json"):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    header = ("case", "aspect", "bbox acc", "bbox unk", "bbox med", "bbox p10",
              "up2x acc", "up2x unk", "up2x med", "up2x p10",
              "corr acc", "corr unk", "corr med", "corr p10", "keep acc", "keep unk", "keep med", "keep p10")
    print(f"{header[0]:<22}{header[1]:>8}" + "".join(f"{name:>10}" for name in header[2:]))
    for name, value in report["summary"].items():
        aspect = value.get("outline_aspect_median")
        row = [f"{name:<22}", f"{aspect if aspect is None else round(aspect, 2):>8}"]
        for variant in ("bbox", "upscaled", "corrected", "kept_aspect"):
            entry = value.get(variant) or {}
            if not entry:
                row.append(f"{'n/a':>10}" * 4)
                continue
            row.extend([f"{round(entry['accuracy'], 3):>10}", f"{entry['unknown']:>10}",
                        f"{round(entry['score_median'], 3):>10}", f"{round(entry['score_p10'], 3):>10}"])
        print("".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
