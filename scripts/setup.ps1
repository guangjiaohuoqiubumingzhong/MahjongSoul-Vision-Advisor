param(
    [string]$PythonExe = 'D:\anaconda3\python.exe',
    [string]$VsDevCmd = 'D:\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat',
    [string]$VenvPath = '.venv'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ($VenvPath -ne '.venv') { throw '本阶段固定使用项目 .venv，防止污染已有 Python 环境。' }
Push-Location $projectRoot
try {
    $sources = Get-Content -LiteralPath configs/sources.json -Raw | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw '用 -PythonExe 指定已安装的 Python 3.12.7。' }
    $version = & $PythonExe -c 'import platform; print(platform.python_version())'
    if ($version -ne '3.12.7') { throw "可复现构建要求 Python 3.12.7；检测到 $version。" }
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        & $PythonExe -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败。' }
    }
    $venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
    & $venvPython -m pip install -r requirements.lock
    if ($LASTEXITCODE -ne 0) { throw '安装依赖失败。' }
    New-Item -ItemType Directory -Force '.tools/downloads' | Out-Null
    $ocrWheel = '.tools/downloads/rapidocr_onnxruntime-1.4.4-py3-none-any.whl'
    if (-not (Test-Path -LiteralPath $ocrWheel)) {
        & $venvPython -m pip download --no-deps --only-binary=:all: --dest '.tools/downloads' 'rapidocr-onnxruntime==1.4.4'
        if ($LASTEXITCODE -ne 0) { throw 'OCR 模型分发包下载失败。' }
    }
    & $venvPython scripts/prepare_ocr.py
    if ($LASTEXITCODE -ne 0) { throw 'OCR 模型校验/提取失败。' }
    foreach ($repo in @(@{Path='third_party/Mortal'; Lock=$sources.mortal}, @{Path='third_party/mjai-reviewer'; Lock=$sources.mjai_reviewer})) {
        if (-not (Test-Path -LiteralPath $repo.Path)) {
            & git clone $repo.Lock.url $repo.Path
            if ($LASTEXITCODE -ne 0) { throw '下载源码失败。' }
            & git -C $repo.Path checkout --detach $repo.Lock.commit
            if ($LASTEXITCODE -ne 0) { throw '固定源码版本失败。' }
        }
        $head = & git -C $repo.Path rev-parse HEAD
        if ($head -ne $repo.Lock.commit) { throw "已存在源码版本不匹配：$($repo.Path)" }
    }
    New-Item -ItemType Directory -Force .tools,models | Out-Null
    $manifest = Get-Content -LiteralPath models/manifest.json -Raw | ConvertFrom-Json
    foreach ($item in $manifest.files.PSObject.Properties) {
        $path = Join-Path 'models' $item.Name
        if (-not (Test-Path -LiteralPath $path)) { Invoke-WebRequest $item.Value.url -OutFile $path }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $item.Value.sha256) { throw "模型来源校验失败：$path" }
    }
    $installer = '.tools/rustup-init.exe'
    if (-not (Test-Path -LiteralPath $installer)) { Invoke-WebRequest $sources.rustup.url -OutFile $installer }
    if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLower() -ne $sources.rustup.sha256) { throw 'Rust installer 校验失败。' }
    $env:RUSTUP_HOME = Join-Path $projectRoot '.tools/rustup'
    $env:CARGO_HOME = Join-Path $projectRoot '.tools/cargo'
    & $installer -y --no-modify-path --profile minimal --default-toolchain 1.90.0-x86_64-pc-windows-msvc
    if ($LASTEXITCODE -ne 0) { throw 'Rust 安装失败。' }
    if (-not (Test-Path -LiteralPath '.tools/windows-sdk/c/um/x64/kernel32.lib')) {
        $zip = '.tools/sdk-x64.zip'
        Invoke-WebRequest $sources.windows_sdk.url -OutFile $zip
        if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower() -ne $sources.windows_sdk.sha256) { throw 'Windows SDK 校验失败。' }
        Expand-Archive -LiteralPath $zip -DestinationPath '.tools/windows-sdk' -Force
    }
    & "$PSScriptRoot/build.ps1" -VsDevCmd $VsDevCmd
    & "$PSScriptRoot/convert.ps1" -InputFile 'data/replays/majsoul-gold-south.export.json' -OutputFile "$projectRoot/data/fixtures/majsoul-gold-south.mjai.jsonl"
    & "$PSScriptRoot/convert.ps1" -InputFile 'data/replays/tenhou-hanchan.export.json' -OutputFile "$projectRoot/data/fixtures/tenhou-hanchan.mjai.jsonl"
    & $venvPython scripts/generate_fixtures.py
    & $venvPython scripts/inspect_model.py
    if ($LASTEXITCODE -ne 0) { throw 'ONNX 图结构/FP32 检查失败。' }
    & $venvPython -m app.cli doctor
    if ($LASTEXITCODE -ne 0) { throw 'CUDA 实际执行验证失败。' }
} finally { Pop-Location }
