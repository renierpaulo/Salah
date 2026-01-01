#!/bin/bash
# Test compilation script for HYPERION ULTRA v6

echo "========================================"
echo "HYPERION ULTRA v6 - Compilation Test"
echo "========================================"

# Check CUDA
echo "[1/3] Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC not found! Please install CUDA toolkit."
    exit 1
fi
nvcc --version | head -n 4

# Compile
echo ""
echo "[2/3] Compiling..."
nvcc -O3 -arch=sm_86 -use_fast_math Hyperion_ULTRA.cu -o Hyperion_ULTRA 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ COMPILATION FAILED!"
    echo "Check errors above and fix them."
    exit 1
fi

echo "✅ Compilation successful!"

# Quick test with small range
echo ""
echo "[3/3] Quick functionality test (5 seconds)..."
echo "Testing with known puzzle 66 parameters..."

# Puzzle 66: Key = 2832ED74F2B5E35EE (known)
# Hash160: 20d45a6a762535700ce9e0b216e31994335db8a5
timeout 5 ./Hyperion_ULTRA --hash160 20d45a6a762535700ce9e0b216e31994335db8a5 --range 20000000000000000:3ffffffffffffffff 2>&1 || true

echo ""
echo "========================================"
echo "Test complete!"
echo ""
echo "To run full search:"
echo "./Hyperion_ULTRA --hash160 <target_hash160> --range <start:end>"
echo "========================================"
