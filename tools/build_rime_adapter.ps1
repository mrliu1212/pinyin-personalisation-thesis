param(
    [Parameter(Mandatory = $true)]
    [string]$RimePrefix
)

$ErrorActionPreference = "Stop"

$prefix = (Resolve-Path -LiteralPath $RimePrefix).Path
$header = Join-Path $prefix "include\rime_api.h"
$library = Join-Path $prefix "lib\rime.lib"
$runtimeDirectories = @(
    (Join-Path $prefix "bin"),
    (Join-Path $prefix "lib")
)

if (-not (Test-Path -LiteralPath $header)) {
    throw "librime header not found: $header"
}
if (-not (Test-Path -LiteralPath $library)) {
    throw "librime import library not found: $library"
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe not found; run this from an x64 Native Tools Command Prompt for Visual Studio 2022"
}

$buildDirectory = Join-Path $PSScriptRoot "..\.build"
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
$source = Join-Path $PSScriptRoot "rime_candidate_cli.cc"
$output = Join-Path $buildDirectory "rime_candidate_cli.exe"
$object = Join-Path $buildDirectory "rime_candidate_cli.obj"

& cl.exe /nologo /std:c++17 /O2 /EHsc /utf-8 "/I$($prefix)\include" "/Fo:$object" $source /link "/LIBPATH:$($prefix)\lib" rime.lib "/OUT:$output"
if ($LASTEXITCODE -ne 0) {
    throw "failed to build rime_candidate_cli.exe"
}

$runtimeLibraries = $runtimeDirectories |
    Where-Object { Test-Path -LiteralPath $_ } |
    ForEach-Object { Get-ChildItem -LiteralPath $_ -Filter "*.dll" }
if (-not $runtimeLibraries) {
    throw "no librime runtime DLLs found under $prefix\bin or $prefix\lib"
}
Copy-Item -LiteralPath $runtimeLibraries.FullName -Destination $buildDirectory -Force
Write-Output "Built $output and copied librime runtime DLLs beside it"
