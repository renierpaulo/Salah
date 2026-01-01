#!/bin/bash
# Salah - HYPERION ULTRA Runner
# Target: f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8
# Range: 400000000000000000:7fffffffffffffffff

# Compile
nvcc -O3 -arch=sm_86 -use_fast_math Hyperion_ULTRA.cu -o Hyperion_ULTRA

# Run
./Hyperion_ULTRA --hash160 f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8 --range 400000000000000000:7fffffffffffffffff
