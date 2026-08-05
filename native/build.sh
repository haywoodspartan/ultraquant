#!/usr/bin/env bash
# UltraQuant native accelerators for Linux / macOS.
#
# The Windows equivalent is native/build.ps1; the two produce libraries with the
# same exports and the same wire format, so the Python tiers do not care which
# built them. Nothing here is required: without these the pure-Python tier runs
# everything with identical results, only slower.
#
# Usage: native/build.sh [--target cpu|cuda|forge|forge-cuda|all]
set -euo pipefail

TARGET="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../ultraquant/native/_bin"
mkdir -p "$OUT"

case "$(uname -s)" in
    Darwin) EXT="dylib"; SOFLAGS="-dynamiclib" ;;
    *)      EXT="so";    SOFLAGS="-shared -fPIC" ;;
esac

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -std=c++17 -Wall -fvisibility=hidden}"

build_cpp() {   # <source> <stem>
    local src="$HERE/$1" out="$OUT/lib$2.$EXT"
    if [ ! -f "$src" ]; then echo "  skip $2 (no $1)"; return; fi
    echo "  $CXX -> $(basename "$out")"
    # shellcheck disable=SC2086
    "$CXX" $CXXFLAGS $SOFLAGS "$src" -o "$out"
}

build_cuda() {  # <source> <stem>
    local src="$HERE/$1" out="$OUT/lib$2.$EXT"
    if [ ! -f "$src" ]; then echo "  skip $2 (no $1)"; return; fi
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "  skip $2 (no nvcc on PATH)"; return
    fi
    local arch="${CUDA_ARCH:-sm_89}"
    echo "  nvcc -arch=$arch -> $(basename "$out")"
    nvcc -O3 -arch="$arch" --shared -Xcompiler -fPIC "$src" -o "$out"
}

case "$TARGET" in
    cpu)        build_cpp uq_core.cpp  ultraquant_native ;;
    cuda)       build_cuda uq_cuda.cu  ultraquant_cuda ;;
    forge)      build_cpp uq_forge.cpp ultraquant_forge ;;
    forge-cuda) build_cuda uq_forge.cu ultraquant_forge_cuda ;;
    all)
        build_cpp  uq_core.cpp  ultraquant_native
        build_cpp  uq_forge.cpp ultraquant_forge
        build_cuda uq_cuda.cu   ultraquant_cuda
        build_cuda uq_forge.cu  ultraquant_forge_cuda
        ;;
    *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac

echo "done -> $OUT"
