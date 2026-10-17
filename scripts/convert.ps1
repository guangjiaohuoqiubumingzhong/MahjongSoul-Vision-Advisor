param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [Parameter(Mandatory=$true)][string]$OutputFile
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $projectRoot 'tools/convlog-cli/target/release/majan-convert.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw '先运行 scripts/build.ps1 构建转换器。' }
$output = & $exe $InputFile
if ($LASTEXITCODE -ne 0) { throw '牌谱转换失败，不写入输出文件。' }
[IO.File]::WriteAllLines($OutputFile, [string[]]$output, [Text.UTF8Encoding]::new($false))
