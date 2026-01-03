#!/usr/bin/env bash
set -euo pipefail

ARCH="${ARCH:-sm_86}"
MAXRREG="${MAXRREG:-96}"

nvcc -O3 -lineinfo -Xptxas=-v --use_fast_math \
  -arch="${ARCH}" \
  -maxrregcount="${MAXRREG}" \
  -DPROFILER_ECC_ONLY=1 \
  -DECC_ONLY_PERSIST_JACOBIAN=0 \
  -I. \
  Hyperion_Profiler_Genius.cu \
  -o Hyperion_Profiler_Genius_fast

echo "Built: ./Hyperion_Profiler_Genius_fast (ARCH=${ARCH}, MAXRREG=${MAXRREG})"
