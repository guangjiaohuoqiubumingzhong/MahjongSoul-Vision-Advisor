import json
import importlib.util
from pathlib import Path
import platform
import os
import subprocess
import sys

from hashlib import sha256

root = Path(__file__).resolve().parents[1]
sources = json.loads((root / "configs/sources.json").read_text(encoding="utf-8"))
actual = subprocess.check_output(["git", "-C", str(root / "third_party/Mortal"), "rev-parse", "HEAD"], text=True).strip()
assert actual == sources["mortal"]["commit"], "Mortal source commit mismatch"
from apply_mortal_patch import apply
snapshot_overlay = apply(check_only=True)
converter = subprocess.check_output(["git", "-C", str(root / "third_party/mjai-reviewer"), "rev-parse", "HEAD"], text=True).strip()
assert converter == sources["mjai_reviewer"]["commit"], "converter source commit mismatch"
assert not subprocess.check_output(["git", "-C", str(root / "third_party/mjai-reviewer"), "status", "--porcelain"], text=True).strip(), "converter source modified"
extension = Path(importlib.util.find_spec("libriichi").origin)
build = {"python": platform.python_version(), "python_architecture": platform.machine(),
         "mortal_commit": actual, "mjai_reviewer_commit": converter,
         "libriichi_sha256": sha256(extension.read_bytes()).hexdigest(),
         "rust": subprocess.check_output([str(root / ".tools/cargo/bin/rustc.exe"), "--version"], text=True,
                   env={**os.environ,"RUSTUP_HOME":str(root/".tools/rustup"),"CARGO_HOME":str(root/".tools/cargo")}).strip(),
         "features": ["pymod"], "default_features": False,
         "sdk_version": sources["windows_sdk"]["version"], "snapshot_overlay": snapshot_overlay}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "configs/build.json"
output.write_text(json.dumps(build, indent=2), encoding="utf-8")
print("Build provenance recorded")
