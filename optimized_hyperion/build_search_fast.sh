#!/bin/bash
# Build script for Hyperion_Search_Fast (batch inversion pipeline)

ARCH=${1:-sm_86}
MAXRREG=${2:-80}

echo "Building Hyperion_Search_Fast with ARCH=$ARCH, MAXRREG=$MAXRREG"

nvcc -O3 -lineinfo \
  -Xptxas=-v \
  --use_fast_math \
  -arch=$ARCH \
  -maxrregcount=$MAXRREG \
  Hyperion_Search_Fast.cu \
  -o Hyperion_Search_Fast

if [ $? -eq 0 ]; then
    echo "✓ Build successful: Hyperion_Search_Fast"
    ls -lh Hyperion_Search_Fast
else
    echo "❌ Build failed"
    exit 1
fi
