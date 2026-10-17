"""Install/check the documented snapshot overlay on the pinned upstream tree."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "patches/mortal-snapshot/manifest.json"


def apply(check_only=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    repo = ROOT / "third_party/Mortal"
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != manifest["source_commit"]:
        raise ValueError("Snapshot overlay requires the pinned Mortal commit")
    for relative, info in manifest["files"].items():
        target = repo / relative
        replacement = MANIFEST.parent / info["overlay"]
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        if digest(replacement) != info["sha256"]:
            raise ValueError(f"Snapshot overlay hash mismatch: {relative}")
        if check_only:
            if digest(target) != info["sha256"]:
                raise ValueError(f"Snapshot source hash mismatch: {relative}")
        elif digest(target) not in (info["original_sha256"], info.get("original_lf_sha256"), info["sha256"]):
            raise ValueError(f"Refusing to overwrite other local edits: {relative}")
        else:
            target.write_bytes(replacement.read_bytes())
    changed = subprocess.check_output(["git", "-C", str(repo), "diff", "--name-only", "HEAD"], text=True).splitlines()
    untracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    if not set(changed + untracked) <= set(manifest["files"]):
        raise ValueError("Unexpected Mortal source changes outside snapshot overlay")
    return {"manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "files": {f"third_party/Mortal/{name}": entry["sha256"] for name, entry in manifest["files"].items()}}


if __name__ == "__main__":
    print(json.dumps(apply("--check" in sys.argv), indent=2))
