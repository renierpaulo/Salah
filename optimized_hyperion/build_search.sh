#!/bin/bash
# Build script for Hyperion_Search (full pipeline with bloom filter)

ARCH=${1:-sm_86}
MAXRREG=${2:-80}

echo "Building Hyperion_Search with ARCH=$ARCH, MAXRREG=$MAXRREG"

nvcc -O3 -lineinfo \
  -Xptxas=-v \
  --use_fast_math \
  -arch=$ARCH \
  -maxrregcount=$MAXRREG \
  Hyperion_Search.cu \
  -o Hyperion_Search_fast

if [ $? -eq 0 ]; then
    echo "✓ Build successful: Hyperion_Search_fast"
    ls -lh Hyperion_Search_fast
else
    echo "❌ Build failed"
    exit 1
fi
