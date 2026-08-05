# UltraQuant native tier build script.
# Usage: powershell -ExecutionPolicy Bypass -File native\build.ps1 [-Target cpu|cuda|all]
# CPU DLL:  MSVC 14.44 (default vcvars64)      -> ultraquant/native/_bin/ultraquant_native.dll
# CUDA DLL: nvcc (CUDA 12.6) + MSVC 14.38      -> ultraquant/native/_bin/ultraquant_cuda.dll
param(
    [ValidateSet('cpu', 'cuda', 'forge', 'forge-cuda', 'all')]
    [string]$Target = 'all'
)
$ErrorActionPreference = 'Stop'

$nativeDir = Split-Path -Parent $PSCommandPath
$projRoot  = Split-Path -Parent $nativeDir
$binDir    = Join-Path $projRoot 'ultraquant\native\_bin'
$buildDir  = Join-Path $nativeDir '_build'
New-Item -ItemType Directory -Force -Path $binDir   | Out-Null
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$failed = $false

if ($Target -in @('cpu', 'all')) {
    Write-Host '=== Building CPU tier (uq_core.cpp -> ultraquant_native.dll) ==='
    $src = Join-Path $nativeDir 'uq_core.cpp'
    $out = Join-Path $binDir 'ultraquant_native.dll'
    Push-Location $buildDir
    try {
        cmd /c "call `"$vcvars`" >nul 2>&1 && cl /nologo /O2 /EHsc /std:c++17 /W3 /LD `"$src`" /Fe:`"$out`""
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { $failed = $true; Write-Host 'CPU build FAILED' }
        else { Write-Host "CPU build OK: $out" }
    } finally { Pop-Location }
}

if ($Target -in @('cuda', 'all')) {
    Write-Host '=== Building CUDA tier (uq_cuda.cu -> ultraquant_cuda.dll) ==='
    $src = Join-Path $nativeDir 'uq_cuda.cu'
    $out = Join-Path $binDir 'ultraquant_cuda.dll'
    Push-Location $buildDir
    try {
        # vcvars_ver=14.38: CUDA 12.6 rejects MSVC 14.4x as a host compiler.
        cmd /c "call `"$vcvars`" -vcvars_ver=14.38 >nul 2>&1 && nvcc -O3 -arch=sm_89 --shared -o `"$out`" `"$src`""
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { $failed = $true; Write-Host 'CUDA build FAILED' }
        else { Write-Host "CUDA build OK: $out" }
    } finally { Pop-Location }
}

if ($Target -in @('forge', 'all')) {
    Write-Host '=== Building forge CPU tier (uq_forge.cpp -> ultraquant_forge.dll) ==='
    $src = Join-Path $nativeDir 'uq_forge.cpp'
    $out = Join-Path $binDir 'ultraquant_forge.dll'
    Push-Location $buildDir
    try {
        cmd /c "call `"$vcvars`" >nul 2>&1 && cl /nologo /O2 /EHsc /std:c++17 /W3 /LD `"$src`" /Fe:`"$out`""
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { $failed = $true; Write-Host 'Forge CPU build FAILED' }
        else { Write-Host "Forge CPU build OK: $out" }
    } finally { Pop-Location }
}

if ($Target -in @('forge-cuda', 'all')) {
    Write-Host '=== Building forge CUDA tier (uq_forge.cu -> ultraquant_forge_cuda.dll) ==='
    $src = Join-Path $nativeDir 'uq_forge.cu'
    $out = Join-Path $binDir 'ultraquant_forge_cuda.dll'
    Push-Location $buildDir
    try {
        cmd /c "call `"$vcvars`" -vcvars_ver=14.38 >nul 2>&1 && nvcc -O3 -arch=sm_89 --shared -o `"$out`" `"$src`""
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { $failed = $true; Write-Host 'Forge CUDA build FAILED' }
        else { Write-Host "Forge CUDA build OK: $out" }
    } finally { Pop-Location }
}

if ($failed) { exit 1 }
Write-Host 'Build complete.'
exit 0
