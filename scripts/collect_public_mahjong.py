"""Download public Mahjong Soul screenshots and import labelled tile crops.

Only public images and provenance are saved; upstream programs are not run.
Existing originals are retained. Rerunning fills missing downloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/public-mahjong"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Majan local screenshot corpus)"}


def fetch(url, path, limit=20 * 1024 * 1024):
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=40) as response:
                with temporary.open("wb") as stream:
                    size = 0
                    while chunk := response.read(256 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("Image exceeds the per-file size limit")
                        stream.write(chunk)
            temporary.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def inspect_image(path):
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return {"width": size[0], "height": size[1], "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ScreenshotCards(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards = []
        self.current = None

    def handle_starttag(self, tag, attributes):
        attr = dict(attributes)
        if tag == "div" and re.fullmatch(r"apphub_Card_\d+", attr.get("id", "")):
            self.current = None
            if attr.get("data-appid") == "2739990":
                self.current = {"id": attr["data-publishedfileid"],
                                "source_page": attr.get("data-modal-content-url")}
        if tag == "img" and self.current and "apphub_CardContentPreviewImage" in attr.get("class", ""):
            url = urllib.parse.urlsplit(attr.get("src", ""))
            if url.hostname in {"images.steamusercontent.com", "steamuserimages-a.akamaihd.net"} and "/ugc/" in url.path:
                # The page uses CDN resize parameters; the same public object
                # without those parameters is the original full-size image.
                self.current["image_url"] = urllib.parse.urlunsplit((url.scheme, url.netloc, url.path, "", ""))
                self.cards.append(self.current)
                self.current = None


def steam(pages, workers):
    cards = {}

    def page(number):
        url = f"https://steamcommunity.com/app/2739990/screenshots/?p={number}&browsefilter=mostrecent"
        path = BASE / f"steam-community/source/pages/recent-{number:03d}.html"
        fetch(url, path, limit=4 * 1024 * 1024)
        parser = ScreenshotCards()
        parser.feed(path.read_text(encoding="utf-8"))
        return parser.cards

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task in as_completed([pool.submit(page, number) for number in range(1, pages + 1)]):
            for card in task.result():
                cards[card["id"]] = card
    records, errors = [], []

    def image(card):
        path = BASE / f"steam-community/images/{card['id']}.jpg"
        fetch(card["image_url"], path)
        return {**card, **inspect_image(path), "source": "steam-community", "kind": "screenshot_candidate",
                "path": path.relative_to(BASE).as_posix(), "upstream_label": None,
                "license": "Public community screenshot; no open reuse licence declared",
                "ground_truth_reviewed": False}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(image, card): card for card in cards.values()}
        for number, task in enumerate(as_completed(jobs), 1):
            try:
                records.append(task.result())
            except Exception as exc:
                errors.append({"source": "steam-community", "id": jobs[task]["id"], "error": str(exc)})
            if number % 20 == 0:
                print(f"Steam: {number}/{len(jobs)}, saved={len(records)}, errors={len(errors)}", flush=True)
    return records, errors


def cvmaj():
    discovery = BASE / "discovery/xdedss__cvmaj.json"
    data = json.loads(discovery.read_text(encoding="utf-8"))
    revision = data["tree"]["sha"]
    source = BASE / "cvmaj/source"
    source.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "readme.md", "samples/readme.md"):
        fetch(f"https://raw.githubusercontent.com/xdedss/cvmaj/{revision}/{name}", source / name.replace("/", "__"))
    archive = source / "pack.zip"
    fetch(f"https://raw.githubusercontent.com/xdedss/cvmaj/{revision}/samples/pack.zip", archive)
    records = []
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            name = PurePosixPath(item.filename)
            if item.is_dir() or name.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not item.filename.startswith("tiles/labeled/label/"):
                continue  # The big/small folders duplicate the labelled samples.
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or any(":" in p for p in name.parts):
                raise ValueError("Unsafe archive entry")
            path = BASE / "cvmaj/images" / Path(*name.parts[3:])
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(zipped.read(item))
            records.append({**inspect_image(path), "source": "cvmaj", "kind": "tile_crop",
                            "path": path.relative_to(BASE).as_posix(), "upstream_label": name.parts[3],
                            "source_page": f"https://github.com/xdedss/cvmaj/tree/{revision}/samples",
                            "archive_member": item.filename, "license": "MIT",
                            "ground_truth_reviewed": False})
    source.joinpath("provenance.json").write_text(json.dumps({"repository": "xdedss/cvmaj",
        "revision": revision, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "imported_folder": "tiles/labeled/label", "excluded": "Duplicate big/small folders and upstream Python code"},
        indent=2), encoding="utf-8")
    # The repository also contains intact game screenshots and useful processed
    # examples. Keep the original folder names so those types remain distinguishable.
    for item in data["tree"]["tree"]:
        name = item["path"]
        if not name.startswith(("img/", "templates/")) or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        path = BASE / "cvmaj/examples" / name
        url = f"https://raw.githubusercontent.com/xdedss/cvmaj/{revision}/{name}"
        fetch(url, path)
        records.append({**inspect_image(path), "source": "cvmaj", "kind": "screenshot_or_processed_example",
                        "path": path.relative_to(BASE).as_posix(), "upstream_label": None,
                        "source_page": f"https://github.com/xdedss/cvmaj/blob/{revision}/{name}",
                        "image_url": url, "license": "MIT", "ground_truth_reviewed": False})
    return records


def pjura():
    checkout = ROOT / ".tools/public-datasets/pjura-tiles"
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    source = BASE / "pjura/source"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkout / "README.md", source / "README.md")
    records = []
    for original in sorted(checkout.joinpath("dataset").rglob("*")):
        if original.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        details = inspect_image(original)  # Reject unfinished Git LFS pointers.
        relative = original.relative_to(checkout)
        path = BASE / "pjura" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(original, path)
        records.append({**details, "source": "pjura", "kind": "tile_crop",
                        "path": path.relative_to(BASE).as_posix(), "upstream_label": original.parent.name,
                        "upstream_split": relative.parts[1], "license": "Apache-2.0 (dataset card)",
                        "source_page": "https://huggingface.co/datasets/pjura/mahjong_souls_tiles",
                        "upstream_path": relative.as_posix(), "revision": revision,
                        "ground_truth_reviewed": False})
    source.joinpath("provenance.json").write_text(json.dumps({"repository": "pjura/mahjong_souls_tiles",
        "revision": revision, "licence_declaration": "README.md", "upstream_images": len(records),
        "relationship": "Same publisher as the project's classifier: debugging data, not an independent score"},
        indent=2), encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steam-pages", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sources", nargs="+", choices=["steam", "cvmaj", "pjura"], default=["steam", "cvmaj", "pjura"])
    args = parser.parse_args()
    records, errors = [], []
    if "steam" in args.sources:
        found, errors = steam(args.steam_pages, args.workers)
        records.extend(found)
    if "cvmaj" in args.sources:
        records.extend(cvmaj())
    if "pjura" in args.sources:
        records.extend(pjura())
    existing = BASE / "manifest.jsonl"
    previous = {r["path"]: r for r in (json.loads(line) for line in existing.read_text(encoding="utf-8").splitlines())} if existing.exists() else {}
    for record in records:
        previous[record["path"]] = {**previous.get(record["path"], {}), **record}
    records = sorted(previous.values(), key=lambda record: record["path"])
    hashes = {}
    for record in records:
        duplicate = hashes.get(record["sha256"])
        record["duplicate_of"] = duplicate
        hashes.setdefault(record["sha256"], record["path"])
    BASE.mkdir(parents=True, exist_ok=True)
    existing.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    summary = {"collected_at": datetime.now(timezone.utc).isoformat(), "images": len(records),
               "unique_images": len(hashes), "by_source": dict(Counter(r["source"] for r in records)),
               "by_kind": dict(Counter(r["kind"] for r in records)),
               "unique_by_source": dict(Counter(r["source"] for r in records if not r["duplicate_of"])),
               "total_bytes": sum(r["bytes"] for r in records), "errors": errors}
    BASE.joinpath("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
