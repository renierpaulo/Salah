#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build_fast.sh                # defaults
#   ./build_fast.sh sm_86 96       # args override
#   ARCH=sm_86 MAXRREG=96 ./build_fast.sh

ARCH_ARG="${1:-}"
MAXRREG_ARG="${2:-}"

ARCH="${ARCH_ARG:-${ARCH:-sm_86}}"
MAXRREG="${MAXRREG_ARG:-${MAXRREG:-96}}"

echo "[build_fast] ARCH=${ARCH} MAXRREG=${MAXRREG}"

nvcc -O3 -lineinfo -Xptxas=-v --use_fast_math \
  -arch="${ARCH}" \
  -maxrregcount="${MAXRREG}" \
  -DPROFILER_ECC_ONLY=1 \
  -DECC_ONLY_PERSIST_JACOBIAN=0 \
  -I. \
  Hyperion_Profiler_Genius.cu \
  -o Hyperion_Profiler_Genius_fast

echo "Built: ./Hyperion_Profiler_Genius_fast (ARCH=${ARCH}, MAXRREG=${MAXRREG})"
