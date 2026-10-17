param(
    [string]$PythonExe = 'D:\anaconda3\python.exe',
    [string]$VsDevCmd = 'D:\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    # Fresh directory on every run. No existing environment or build cache is used.
    $checkRoot = Join-Path $projectRoot ('.tools/clean-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $checkRoot | Out-Null
    $venv = Join-Path $checkRoot 'venv'
    & $PythonExe -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw '干净环境创建失败。' }
    $freshPython = Join-Path $venv 'Scripts/python.exe'
    & $freshPython -m pip install -r requirements.lock
    if ($LASTEXITCODE -ne 0) { throw '干净环境依赖安装失败。' }
    $env:RUSTUP_HOME = Join-Path $projectRoot '.tools/rustup'
    $env:CARGO_HOME = Join-Path $projectRoot '.tools/cargo'
    $env:PATH = (Join-Path $env:CARGO_HOME 'bin') + ';' + $env:PATH
    $env:PYO3_PYTHON = $freshPython
    & $freshPython scripts/apply_mortal_patch.py
    if ($LASTEXITCODE -ne 0) { throw '当前局面适配补丁校验失败。' }
    $env:RUSTFLAGS = $null
    $sdkUm = Join-Path $projectRoot '.tools/windows-sdk/c/um/x64'
    $sdkUcrt = Join-Path $projectRoot '.tools/windows-sdk/c/ucrt/x64'
    $env:CARGO_ENCODED_RUSTFLAGS = [string]::Join([char]0x1f, @('-L', "native=$sdkUm", '-L', "native=$sdkUcrt"))
    $env:CARGO_TARGET_DIR = Join-Path $checkRoot 'rust-target'
    & cmd.exe /d /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 && cargo build --manifest-path third_party/Mortal/Cargo.toml -p libriichi --lib --release --locked --no-default-features --features pymod"
    if ($LASTEXITCODE -ne 0) { throw '干净构建失败。' }
    Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR 'release/riichi.dll') -Destination (Join-Path $venv 'Lib/site-packages/libriichi.pyd')
    $buildManifest = Join-Path $checkRoot 'build.json'
    & $freshPython scripts/record_build.py $buildManifest
    if ($LASTEXITCODE -ne 0) { throw '干净构建来源记录失败。' }
    $env:MAJAN_CLEAN_BUILD = $buildManifest
    & $freshPython -c 'import json,os,pathlib; from app.engine.runtime import OnnxRuntime,ROOT; config=json.loads((ROOT/"configs/model.json").read_text(encoding="utf-8")); config["build_manifest_path"]=os.environ["MAJAN_CLEAN_BUILD"]; path=pathlib.Path(os.environ["MAJAN_CLEAN_BUILD"]).with_name("model.json"); path.write_text(json.dumps(config),encoding="utf-8"); runtime=OnnxRuntime(path); report={"status":"passed","clean_build":os.environ["MAJAN_CLEAN_BUILD"],"interface":runtime.interface,**runtime.verification}; (ROOT/"reports/clean-environment.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print("Clean environment: GPU inference verified")'
    if ($LASTEXITCODE -ne 0) { throw '干净环境 CUDA 验收失败。' }
} finally {
    $env:CARGO_TARGET_DIR = $null
    $env:MAJAN_CLEAN_BUILD = $null
    Pop-Location
}
