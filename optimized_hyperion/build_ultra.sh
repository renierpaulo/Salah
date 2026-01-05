#!/bin/bash
ARCH=${1:-sm_86}
MAXRREG=${2:-80}
echo "Building Hyperion_Search_Ultra with ARCH=$ARCH, MAXRREG=$MAXRREG"
nvcc -O3 -lineinfo -Xptxas=-v --use_fast_math -arch=$ARCH -maxrregcount=$MAXRREG Hyperion_Search_Ultra.cu -o Hyperion_Search_Ultra
if [ $? -eq 0 ]; then
    echo "✓ Build successful: Hyperion_Search_Ultra"
    ls -lh Hyperion_Search_Ultra
else
    echo "❌ Build failed"
    exit 1
fi
