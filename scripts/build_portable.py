"""Produce an allowlisted Windows/NVIDIA onedir release and a ZIP archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".tools/portable-stage"
DIST = ROOT / "dist"


def copy(relative):
    source = ROOT / relative
    dest = STAGE / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def manifest(relative):
    copy(relative)
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def stage_resources():
    if STAGE.exists():
        raise RuntimeError(f"暂存目录已存在，请使用 --reuse-stage 或选择新的构建目录：{STAGE}")
    STAGE.mkdir(parents=True)
    config = manifest("configs/window-steam-1366-v2.json")
    for relative in ("configs/build.json", "configs/rules.json", "configs/river-polygons.json",
                     "models/mortal.onnx", "models/model_meta.json", "models/manifest.json",
                     "patches/mortal-snapshot/manifest.json", "THIRD_PARTY_NOTICES.md"):
        copy(relative)
    build = json.loads((ROOT / "configs/build.json").read_text(encoding="utf-8"))
    for relative in build["snapshot_overlay"]["files"]:
        copy(relative)
    model = manifest("configs/model.json")
    model["profile_dir"] = "../userdata/profiles"
    (STAGE / "configs/model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    # Only files referenced by the active template bank are runtime resources.
    relative = Path(config["templates"])
    templates = manifest(relative)
    for item in templates["templates"]:
        copy(relative.parent / item["file"])
    svm_path = Path(config["classifier_manifest"])
    for item in manifest(svm_path)["models"]:
        copy(svm_path.parent / item["path"])
    sideways = Path(config["tile_classifier"]["sideways_verifier_manifest"])
    value = manifest(sideways)
    copy(sideways.parent / value["features"]["path"])
    for item in value["models"]:
        copy(sideways.parent / item["path"])
    active = Path(config["tile_classifier"]["manifest"])
    value = manifest(active)
    copy(active.parent / value["model"])
    optimization = value["optimization"]
    copy(active.parent / optimization["validation_file"])
    original = (active.parent / optimization["source_manifest"])
    source = manifest(original)
    copy(original.parent / source["model"])
    ocr = manifest("assets/ocr/manifest.json")
    copy(Path("assets/ocr") / ocr["model"])
    winds = Path(config["self_wind"]["manifest"])
    for item in manifest(winds)["templates"]:
        copy(winds.parent / item["file"])
    # These are provenance references in the active config, not screenshot data.
    for name in ("meld_source_review", "kan_source_review"):
        if config.get(name):
            copy(config[name])


def licenses_and_source(folder):
    licenses = folder / "licenses"
    licenses.mkdir(exist_ok=True)
    for rel in ("THIRD_PARTY_NOTICES.md", "third_party/Mortal/LICENSE", "models/LICENSE.upstream",
                "models/README.md", "models/pjura-vision/LICENSE-2.0.txt",
                "models/pjura-vision/upstream/README.md", "assets/ocr/LICENSE.upstream"):
        dest = licenses / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    site = ROOT / ".venv/Lib/site-packages"
    for info in site.glob("*.dist-info"):
        if info.name.startswith(("pip-", "pytest-", "iniconfig-", "pluggy-", "pygments-")):
            continue
        for path in info.rglob("*"):
            if path.is_file() and any(word in path.name.lower() for word in ("license", "copying", "notice", "eula")):
                dest = licenses / "python-packages" / path.relative_to(site)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    for path in Path(sys.base_prefix).glob("LICENSE*"):
        shutil.copy2(path, licenses / ("Python-" + path.name))
    # Keep the exact application and patched upstream source with this build.
    with zipfile.ZipFile(licenses / "corresponding-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("app", "scripts", "packaging", "patches", "third_party/Mortal", "tools/convlog-cli", "tests"):
            for path in (ROOT / name).rglob("*"):
                parts = path.relative_to(ROOT / name).parts
                if path.is_file() and not any(p in {".git", "target", "__pycache__", ".pytest_cache"} for p in parts):
                    archive.write(path, path.relative_to(ROOT).as_posix())
        for name in ("launch_portable.pyw", "launch_advice.pyw", "requirements.lock", "pyproject.toml",
                     "README.md", "THIRD_PARTY_NOTICES.md", "configs/sources.json", "configs/build.json"):
            archive.write(ROOT / name, name)


def build_runtime(folder):
    """Keep CPython's normal executable and extension loading for WGC compatibility."""
    if folder.exists():
        raise RuntimeError(f"输出目录已存在：{folder}；请使用 --finalize-only 或更换版本目录。")
    internal = folder / "_internal"
    shutil.copytree(STAGE, internal)
    base = Path(sys.base_prefix)
    for name in ("python.exe", "pythonw.exe", "python312.dll", "python3.dll"):
        shutil.copy2(base / name, internal / name)
    for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
                 "msvcp140_atomic_wait.dll", "concrt140.dll", "vcomp140.dll", "ffi.dll", "tcl86t.dll", "tk86t.dll",
                 "zlib.dll", "libcrypto-3-x64.dll", "libssl-3-x64.dll", "libexpat.dll", "LIBBZ2.dll", "liblzma.dll", "sqlite3.dll"):
        source = next((p for p in (base / name, base / "Library/bin" / name) if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(name)
        shutil.copy2(source, internal / name)
    for source in (base / "DLLs").glob("*.pyd"):
        if not source.name.startswith(("_test", "_ctypes_test", "_xx")):
            shutil.copy2(source, internal / source.name)
    for name, target in (("tcl8.6", "_tcl_data"), ("tk8.6", "_tk_data"), ("tcl8", "tcl8")):
        shutil.copytree(base / "Library/lib" / name, internal / target)
    with zipfile.ZipFile(internal / "python312.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in (base / "Lib").rglob("*.py"):
            relative = source.relative_to(base / "Lib")
            if not set(relative.parts) & {"site-packages", "test", "tests", "__pycache__", "ensurepip", "idlelib", "turtledemo"}:
                archive.write(source, relative.as_posix())
    # _pth enables CPython isolation: ignore system Python installations, user
    # site-packages and PYTHONPATH. Every import comes from these two entries.
    (internal / "python312._pth").write_text("python312.zip\n.\n", encoding="ascii")
    shutil.copytree(ROOT / "app", internal / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (internal / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/verify_window_content_capture.py", internal / "scripts/verify_window_content_capture.py")
    shutil.copy2(ROOT / "launch_portable.pyw", internal / "launch_portable.pyw")
    site = ROOT / ".venv/Lib/site-packages"
    for name in ("numpy", "numpy.libs", "cv2", "PIL", "psutil", "mss", "mahjong", "windows_capture",
                 "onnxruntime", "flatbuffers", "coloredlogs", "humanfriendly", "colorama", "packaging"):
        shutil.copytree(site / name, internal / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "test"))
    for name in ("libriichi.pyd", "typing_extensions.py"):
        shutil.copy2(site / name, internal / name)
    for source in (site / "nvidia").rglob("*"):
        if source.is_file() and source.suffix.lower() in {".dll", ".py"}:
            target = internal / source.relative_to(site)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    distributions = ("numpy", "opencv-python", "Pillow", "psutil", "mss", "mahjong", "windows-capture",
                     "onnxruntime-gpu", "flatbuffers", "coloredlogs", "humanfriendly", "colorama", "packaging", "typing-extensions")
    for distribution in distributions:
        prefix = distribution.replace("-", "_").lower() + "-"
        source = next(p for p in site.glob("*.dist-info") if p.name.lower().startswith(prefix))
        shutil.copytree(source, internal / source.name)
    for source in site.glob("nvidia*.dist-info"):
        shutil.copytree(source, internal / source.name)
    (internal / "portable-runtime.json").write_text(json.dumps({"python": sys.version, "mode": "isolated-cpython"}), encoding="utf-8")
    build_launcher(folder)


def build_launcher(folder):
    rustc = ROOT / ".tools/rustup/toolchains/1.90.0-x86_64-pc-windows-msvc/bin/rustc.exe"
    sdk = ROOT / ".tools/windows-sdk/c"
    output = ROOT / ".tools/portable-launcher"
    output.mkdir(exist_ok=True)
    args = [str(rustc), "--edition=2021", "-O", "-C", "strip=symbols", "-C", "target-feature=+crt-static", "--crate-name", "majan_launcher",
            "-L", "native=" + str(sdk / "um/x64"), "-L", "native=" + str(sdk / "ucrt/x64"),
            str(ROOT / "packaging/launcher.rs"), "-o", str(output / "Majan.exe")]
    subprocess.run(args, cwd=ROOT, check=True)
    shutil.copy2(output / "Majan.exe", folder / "Majan.exe")


def finalize(folder, version="0.1.0-portable-test1"):
    licenses_and_source(folder)
    (folder / "userdata").mkdir(exist_ok=True)
    (folder / "使用说明.txt").write_text(
        "Majan 便携测试版 · Windows x64 / NVIDIA\n\n"
        "1. 将整个压缩包解压到可写文件夹，再双击 Majan.exe；请保留 _internal 文件夹。\n"
        "2. 打开雀魂 Steam 客户端，使用默认牌面；在面板中选择窗口和尺寸，点击“开始自动建议”。\n"
        "3. 本版沿用当前项目功能与识别限制，支持面板中的 1366×768 和 1920×1080 预设。\n"
        "4. 需要兼容 CUDA 12.x 的 NVIDIA 显卡驱动；无需另装 Python、Rust、CUDA Toolkit 或 cuDNN。\n"
        "5. 首次准备模型需要一些时间。设置及运行记录位于 userdata，截图会持续占用磁盘，可停止程序后清理。\n"
        "6. 无法启动时查看 userdata/startup.log；双击“检查运行环境.cmd”后查看 userdata/self-test.json。\n"
        "7. 请完整解压后运行，不要直接在压缩包内启动。测试版尚未经另一台干净电脑验收。\n\n"
        "本机测试限制：独立程序的 Windows.Graphics.Capture 调用返回拒绝访问，现有 PrintWindow 兼容采集已通过测试。\n"
        "请先保持雀魂窗口正常显示且不被遮挡；后台/遮挡采集还需在你的实际运行环境确认。\n\n"
        "第三方许可及对应源码：licenses 文件夹。模型与程序版本：release.json。\n", encoding="utf-8-sig")
    (folder / "检查运行环境.cmd").write_text(
        '@echo off\r\nstart "" /wait "%~dp0Majan.exe" --self-test\r\n'
        'start "" notepad.exe "%~dp0userdata\\self-test.json"\r\n', encoding="ascii")
    release = {"version": version, "built_at_utc": datetime.now(timezone.utc).isoformat(),
               "platform": "Windows-x64-NVIDIA-CUDA12", "python": sys.version,
               "runtime": "isolated-cpython-with-native-launcher",
               "config": json.loads((STAGE / "configs/window-steam-1366-v2.json").read_text(encoding="utf-8"))["version"]}
    files = []
    for path in sorted(folder.rglob("*")):
        if (path.is_file() and path.name != "release.json" and path.suffix not in {".pyc", ".log"}
                and not set(path.relative_to(folder).parts) & {"userdata", "__pycache__"}):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"path": path.relative_to(folder).as_posix(), "size": path.stat().st_size, "sha256": digest})
    release["files"] = files
    (folder / "release.json").write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    global STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0-portable-test1")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    parser.add_argument("--output-dir", type=Path, default=DIST / "Majan-portable")
    parser.add_argument("--reuse-stage", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--zip-only", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--launcher-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("Version must contain only letters, digits, dots, underscores and hyphens")
    STAGE = (ROOT / args.stage_dir).resolve()
    folder = (ROOT / args.output_dir).resolve()
    if not STAGE.is_relative_to(ROOT) or not folder.is_relative_to(ROOT):
        parser.error("Stage and output directories must be inside the project")
    if not args.zip_only:
        if args.launcher_only:
            build_launcher(folder)
        elif not args.finalize_only:
            if not args.reuse_stage:
                stage_resources()
            build_runtime(folder)
        finalize(folder, args.version)
    if args.no_zip:
        return
    archive = DIST / f"Majan-{args.version}-win64-nvidia.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for path in sorted(folder.rglob("*")):
            if (path.is_file() and path.suffix not in {".pyc", ".log"}
                    and not set(path.relative_to(folder).parts) & {"userdata", "__pycache__"}):
                z.write(path, (Path("Majan") / path.relative_to(folder)).as_posix())
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
