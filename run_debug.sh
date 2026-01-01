#!/bin/bash
# Salah - HYPERION ULTRA Debug Runner
# This script tests the program with a known key to verify it works

echo "=========================================="
echo "HYPERION ULTRA - Debug Test Suite"
echo "=========================================="

# Compile with debug symbols
echo "[1/4] Compiling with optimizations..."
nvcc -O3 -arch=sm_86 -use_fast_math -lineinfo Hyperion_ULTRA.cu -o Hyperion_ULTRA_debug 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi
echo "✅ Compilation successful"

# Test 1: Known key close to start (should find quickly)
echo ""
echo "[2/4] Test 1: Key close to range start..."
echo "      Target: Puzzle 71 hash160"
echo "      This should be found within first few batches if key is close"
echo ""

# Using the actual puzzle 71 target
# Hash160: f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8
# Expected key is in range: 400000000000000000:7fffffffffffffffff

./Hyperion_ULTRA_debug --hash160 f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8 --range 400000000000000000:7fffffffffffffffff &
PID=$!

# Let it run for 60 seconds then check
sleep 60

if ps -p $PID > /dev/null; then
    echo ""
    echo "[INFO] Program still running after 60s - checking status..."
    echo "[INFO] If you see 'infinity' or 'ZERO' warnings above, that indicates the bug"
    echo "[INFO] Press Ctrl+C to stop the test"
    
    # Wait for user interrupt or program completion
    wait $PID
else
    echo ""
    echo "[INFO] Program finished"
fi

echo ""
echo "=========================================="
echo "Debug test complete"
echo "=========================================="
