param(
    [switch]$ReuseStage,
    [switch]$NoZip,
    [string]$Version = '0.1.0-portable-test1',
    [string]$StageDirectory = '.tools/portable-stage',
    [string]$OutputDirectory = 'dist/Majan-portable',
    [string]$VsDevCmd = 'D:\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw '请用 -VsDevCmd 指定 Visual Studio C++ Build Tools 的 VsDevCmd.bat。'
}
Push-Location $projectRoot
try {
    $arguments = @()
    if ($Version -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw '版本号格式无效。' }
    $arguments += @('--version', $Version, '--stage-dir', "`"$StageDirectory`"", '--output-dir', "`"$OutputDirectory`"")
    if ($ReuseStage) { $arguments += '--reuse-stage' }
    if ($NoZip) { $arguments += '--no-zip' }
    $python = Join-Path $projectRoot '.venv/Scripts/python.exe'
    $builder = Join-Path $projectRoot 'scripts/build_portable.py'
    & cmd.exe /d /c "`"$VsDevCmd`" -arch=x64 -host_arch=x64 && `"$python`" `"$builder`" $($arguments -join ' ')"
    if ($LASTEXITCODE -ne 0) { throw '便携包构建失败。' }
} finally {
    Pop-Location
}
