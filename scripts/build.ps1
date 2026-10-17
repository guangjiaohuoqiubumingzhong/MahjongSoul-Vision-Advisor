param(
    [string]$VsDevCmd = 'D:\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $VsDevCmd)) {
        throw '需要 Visual Studio 2022 C++ Build Tools；用 -VsDevCmd 指定 VsDevCmd.bat。'
    }
    $env:RUSTUP_HOME = Join-Path $projectRoot '.tools/rustup'
    $env:CARGO_HOME = Join-Path $projectRoot '.tools/cargo'
    $env:PYO3_PYTHON = Join-Path $projectRoot '.venv/Scripts/python.exe'
    & $env:PYO3_PYTHON scripts/apply_mortal_patch.py
    if ($LASTEXITCODE -ne 0) { throw '当前局面适配补丁校验失败。' }
    $env:PATH = (Join-Path $env:CARGO_HOME 'bin') + ';' + $env:PATH
    $sdkUm = Join-Path $projectRoot '.tools/windows-sdk/c/um/x64'
    $sdkUcrt = Join-Path $projectRoot '.tools/windows-sdk/c/ucrt/x64'
    # Encoded flags preserve each argument, including roots containing spaces.
    $env:RUSTFLAGS = $null
    $env:CARGO_ENCODED_RUSTFLAGS = [string]::Join([char]0x1f, @('-L', "native=$sdkUm", '-L', "native=$sdkUcrt"))
    $cargo = Join-Path $env:CARGO_HOME 'bin/cargo.exe'
    foreach ($target in @(
        'build --manifest-path third_party/Mortal/Cargo.toml -p libriichi --lib --release --locked --no-default-features --features pymod',
        'build --manifest-path tools/convlog-cli/Cargo.toml --release --locked'
    )) {
        & cmd.exe /d /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 && cargo $target"
        if ($LASTEXITCODE -ne 0) { throw "Rust 构建失败：$target" }
    }
    Copy-Item -LiteralPath 'third_party/Mortal/target/release/riichi.dll' -Destination '.venv/Lib/site-packages/libriichi.pyd' -Force
    & $env:PYO3_PYTHON -c 'import libriichi; from libriichi.consts import obs_shape; print("libriichi OK", obs_shape(3))'
    if ($LASTEXITCODE -ne 0) { throw 'libriichi 导入失败。' }
    & $env:PYO3_PYTHON scripts/record_build.py
    if ($LASTEXITCODE -ne 0) { throw '构建来源校验失败。' }
} finally { Pop-Location }
