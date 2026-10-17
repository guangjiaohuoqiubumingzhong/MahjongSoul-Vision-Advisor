"""Download the pinned, public tile classifier (not the gated assistant)."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "pjura/mahjong_soul_vision"
REVISION = "ea01ca0bcb9be0fc10a5fa09ee8356770c25adc5"
WEIGHT_SHA256 = "a3cc566c8011b4496239a5228de3985e37ba2c2dfceefd52f67bb34c5f7e49c0"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    folder = ROOT / "models/pjura-vision/upstream"
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("config.json", "preprocessor_config.json", "README.md", "pytorch_model.bin"):
        target = folder / name
        if target.exists() and (name != "pytorch_model.bin" or digest(target) == WEIGHT_SHA256):
            files[name] = digest(target)
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            total = 0
            milestone = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                total += len(block)
                if total // (32 * 1024 * 1024) > milestone:
                    milestone = total // (32 * 1024 * 1024)
                    print(f"{name}: {total // (1024 * 1024)} MiB", flush=True)
        if name == "pytorch_model.bin" and digest(temporary) != WEIGHT_SHA256:
            raise ValueError("Upstream pjura weight SHA256 mismatch")
        temporary.replace(target)
        files[name] = digest(target)
    provenance = {"repo": REPO, "revision": REVISION, "license": "Apache-2.0",
                  "source": f"https://huggingface.co/{REPO}/tree/{REVISION}", "files": files}
    (folder / "download.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
