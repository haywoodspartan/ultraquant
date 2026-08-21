# Builds the native cognitive tier with MSVC.
#
# No external dependencies, matching the pure-Python tier's own rule:
# if the Python side needs no numpy, the native side needs no GMP.
param([string]$Tool = "all")

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $here "_build"
New-Item -ItemType Directory -Force $out | Out-Null

$vs = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vs)) {
    $vs = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
}
if (-not (Test-Path $vs)) { throw "no MSVC found; set up a developer prompt" }

$inc = Join-Path $here "include"
$sources = @(Get-ChildItem (Join-Path $here "src") -Filter *.cpp |
             ForEach-Object { $_.FullName })

$tools = @{
    "bigint_probe" = "uq_bigint_probe.exe"
    "calc_probe"   = "uq_calc.exe"
}

foreach ($name in $tools.Keys) {
    if ($Tool -ne "all" -and $Tool -ne $name) { continue }
    $toolPath = Join-Path $here "tools\$name.cpp"
    if (-not (Test-Path $toolPath)) { continue }
    $exe = Join-Path $out $tools[$name]
    $files = ($sources + $toolPath | ForEach-Object { '"' + $_ + '"' }) -join " "
    $line = '"' + $vs + '" >nul 2>&1 && cl /nologo /std:c++17 /EHsc /O2 /W4 ' +
            '/Zc:__cplusplus /I "' + $inc + '" ' + $files +
            ' /Fe:"' + $exe + '" /Fo:"' + $out + '\"'
    cmd /c $line | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "build failed: $name" }
    Write-Host ("built " + $tools[$name])
}
