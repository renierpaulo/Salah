#!/bin/bash
ARCH=${1:-sm_86}
MAXRREG=${2:-80}
echo "Building Hyperion_Search_Final with ARCH=$ARCH, MAXRREG=$MAXRREG"
nvcc -O3 -lineinfo -Xptxas=-v --use_fast_math -arch=$ARCH -maxrregcount=$MAXRREG Hyperion_Search_Final.cu -o Hyperion_Search_Final
if [ $? -eq 0 ]; then
    echo "✓ Build successful: Hyperion_Search_Final"
    ls -lh Hyperion_Search_Final
else
    echo "❌ Build failed"
    exit 1
fi
