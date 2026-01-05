#!/bin/bash
# Build script for Hyperion_Search_V2 (optimized full pipeline)

ARCH=${1:-sm_86}
MAXRREG=${2:-80}

echo "Building Hyperion_Search_V2 with ARCH=$ARCH, MAXRREG=$MAXRREG"

nvcc -O3 -lineinfo \
  -Xptxas=-v \
  --use_fast_math \
  -arch=$ARCH \
  -maxrregcount=$MAXRREG \
  Hyperion_Search_V2.cu \
  -o Hyperion_Search_V2

if [ $? -eq 0 ]; then
    echo "✓ Build successful: Hyperion_Search_V2"
    ls -lh Hyperion_Search_V2
else
    echo "❌ Build failed"
    exit 1
fi
