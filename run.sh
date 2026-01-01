#!/bin/bash
# Salah - HYPERION ULTRA Runner (v6 + Debug)
# Target: f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8
# Range: 400000000000000000:7fffffffffffffffff

# Compile with optimizations
echo "Compiling HYPERION ULTRA v6..."
nvcc -O3 -arch=sm_86 -use_fast_math -lineinfo Hyperion_ULTRA.cu -o Hyperion_ULTRA

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"

# Run
echo ""
echo "Starting search..."
./Hyperion_ULTRA --hash160 f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8 --range 400000000000000000:7fffffffffffffffff
