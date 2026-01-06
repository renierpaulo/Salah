/*
 * HYPERION MICRO PROFILER - Detailed Operation-Level Analysis
 * 
 * Measures exact cycle count for each micro-operation:
 * - fieldMul (256-bit modular multiplication)
 * - fieldSqr (256-bit modular squaring)
 * - fieldInv (Fermat inversion - 256 squarings + muls)
 * - SHA256 (64 rounds)
 * - RIPEMD160 (80 rounds)
 * 
 * Goal: Identify exact hotspots and optimize each
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 256
#define THREADS_PER_BLOCK 256
#define TEST_ITERATIONS 1000

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

// Profiling counters
__device__ unsigned long long d_cycles_fieldMul = 0;
__device__ unsigned long long d_cycles_fieldSqr = 0;
__device__ unsigned long long d_cycles_fieldSub = 0;
__device__ unsigned long long d_cycles_fieldAdd = 0;
__device__ unsigned long long d_cycles_fieldInv = 0;
__device__ unsigned long long d_cycles_SHA256 = 0;
__device__ unsigned long long d_cycles_RIPEMD160 = 0;
__device__ unsigned long long d_cycles_total = 0;

__device__ unsigned long long d_count_ops = 0;

// Test data
__device__ __constant__ uint64_t TEST_A[4] = {
    0x79BE667EF9DCBBACULL, 0x55A06295CE870B07ULL, 
    0x029BFCDB2DCE28D9ULL, 0x59F2815B16F81798ULL
};
__device__ __constant__ uint64_t TEST_B[4] = {
    0x483ADA7726A3C465ULL, 0x5DA4FBFC0E1108A8ULL,
    0xFD17B448A6855419ULL, 0x9C47D08FFB10D4B8ULL
};

// Micro-profiling kernel
__global__ void kernel_micro_profile() {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    uint64_t a[4], b[4], c[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        a[i] = TEST_A[i] + tid;
        b[i] = TEST_B[i] + tid;
    }
    
    unsigned long long total_start = clock64();
    
    // ============ PROFILE fieldMul ============
    unsigned long long start = clock64();
    #pragma unroll
    for(int i = 0; i < TEST_ITERATIONS; i++) {
        fieldMul(a, b, c);
        a[0] ^= c[0]; // Prevent optimization
    }
    unsigned long long cycles_mul = clock64() - start;
    
    // ============ PROFILE fieldSqr ============
    start = clock64();
    #pragma unroll
    for(int i = 0; i < TEST_ITERATIONS; i++) {
        fieldSqr(a, c);
        a[0] ^= c[0];
    }
    unsigned long long cycles_sqr = clock64() - start;
    
    // ============ PROFILE fieldSub ============
    start = clock64();
    #pragma unroll
    for(int i = 0; i < TEST_ITERATIONS; i++) {
        fieldSub(a, b, c);
        a[0] ^= c[0];
    }
    unsigned long long cycles_sub = clock64() - start;
    
    // ============ PROFILE fieldAdd ============
    start = clock64();
    #pragma unroll
    for(int i = 0; i < TEST_ITERATIONS; i++) {
        fieldAdd(a, b, c);
        a[0] ^= c[0];
    }
    unsigned long long cycles_add = clock64() - start;
    
    // ============ PROFILE fieldInv (Fermat) ============
    start = clock64();
    for(int i = 0; i < 10; i++) { // Only 10 iterations (expensive!)
        fieldInv_Fermat(a, c);
        a[0] ^= c[0];
    }
    unsigned long long cycles_inv = clock64() - start;
    
    // ============ PROFILE SHA256 ============
    uint32_t sha_state[8];
    start = clock64();
    #pragma unroll
    for(int i = 0; i < TEST_ITERATIONS; i++) {
        sha256_opt(a[0], a[1], a[2], a[3], 0x02, sha_state);
        a[0] ^= sha_state[0];
    }
    unsigned long long cycles_sha = clock64() - start;
    
    // ============ PROFILE RIPEMD160 ============
    uint8_t hash[20];
    start = clock64();
    #pragma unroll
    for(int i = 0; i < TEST_ITERATIONS; i++) {
        ripemd160_opt(sha_state, hash);
        a[0] ^= hash[0];
    }
    unsigned long long cycles_ripemd = clock64() - start;
    
    unsigned long long total_end = clock64();
    
    // Atomic add to global counters (only thread 0 of each block)
    if(threadIdx.x == 0) {
        atomicAdd(&d_cycles_fieldMul, cycles_mul);
        atomicAdd(&d_cycles_fieldSqr, cycles_sqr);
        atomicAdd(&d_cycles_fieldSub, cycles_sub);
        atomicAdd(&d_cycles_fieldAdd, cycles_add);
        atomicAdd(&d_cycles_fieldInv, cycles_inv);
        atomicAdd(&d_cycles_SHA256, cycles_sha);
        atomicAdd(&d_cycles_RIPEMD160, cycles_ripemd);
        atomicAdd(&d_cycles_total, total_end - total_start);
        atomicAdd(&d_count_ops, 1);
    }
}

// Full pipeline profiling (simulates actual search)
__global__ void kernel_pipeline_profile(unsigned long long* out_ecc, unsigned long long* out_hash) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    uint64_t base_x[4], base_y[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        base_x[i] = TEST_A[i] + tid;
        base_y[i] = TEST_B[i] + tid;
    }
    
    uint64_t gx[4], gy[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        gx[i] = TEST_B[i];
        gy[i] = TEST_A[i];
    }
    
    unsigned long long ecc_cycles = 0;
    unsigned long long hash_cycles = 0;
    
    for(int iter = 0; iter < 256; iter++) {
        // ECC PART: Point addition with batch inversion simulation
        unsigned long long ecc_start = clock64();
        
        uint64_t dx[4], dy[4], inv_dx[4], lambda[4], lambda_sq[4], temp[4], x3[4], y3[4];
        
        fieldSub(gx, base_x, dx);
        fieldSub(gy, base_y, dy);
        fieldInv_Fermat(dx, inv_dx);  // This is the expensive part!
        fieldMul(dy, inv_dx, lambda);
        fieldSqr(lambda, lambda_sq);
        fieldSub(lambda_sq, base_x, temp);
        fieldSub(temp, gx, x3);
        fieldSub(base_x, x3, temp);
        fieldMul(lambda, temp, y3);
        fieldSub(y3, base_y, y3);
        
        ecc_cycles += clock64() - ecc_start;
        
        // HASH PART: SHA256 + RIPEMD160
        unsigned long long hash_start = clock64();
        
        uint32_t sha[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha);
        
        uint8_t hash160[20];
        ripemd160_opt(sha, hash160);
        
        hash_cycles += clock64() - hash_start;
        
        // Update for next iteration
        base_x[0] ^= x3[0];
        base_y[0] ^= y3[0];
        gx[0]++;
    }
    
    if(threadIdx.x == 0) {
        atomicAdd(out_ecc, ecc_cycles);
        atomicAdd(out_hash, hash_cycles);
    }
}

int main() {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   HYPERION MICRO PROFILER - Operation-Level Analysis        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Configuration:\n");
    printf("  Blocks:     %d\n", NUM_BLOCKS);
    printf("  Threads:    %d\n", THREADS_PER_BLOCK);
    printf("  Iterations: %d per operation\n\n", TEST_ITERATIONS);
    
    // Reset counters
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(d_cycles_fieldMul, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_cycles_fieldSqr, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_cycles_fieldSub, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_cycles_fieldAdd, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_cycles_fieldInv, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_cycles_SHA256, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_cycles_RIPEMD160, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_cycles_total, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_count_ops, &zero, sizeof(zero));
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("PART 1: Individual Operation Profiling\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    kernel_micro_profile<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
    
    // Fetch results
    unsigned long long cycles_mul, cycles_sqr, cycles_sub, cycles_add, cycles_inv, cycles_sha, cycles_ripemd, cycles_total, count;
    cudaMemcpyFromSymbol(&cycles_mul, d_cycles_fieldMul, sizeof(cycles_mul));
    cudaMemcpyFromSymbol(&cycles_sqr, d_cycles_fieldSqr, sizeof(cycles_sqr));
    cudaMemcpyFromSymbol(&cycles_sub, d_cycles_fieldSub, sizeof(cycles_sub));
    cudaMemcpyFromSymbol(&cycles_add, d_cycles_fieldAdd, sizeof(cycles_add));
    cudaMemcpyFromSymbol(&cycles_inv, d_cycles_fieldInv, sizeof(cycles_inv));
    cudaMemcpyFromSymbol(&cycles_sha, d_cycles_SHA256, sizeof(cycles_sha));
    cudaMemcpyFromSymbol(&cycles_ripemd, d_cycles_RIPEMD160, sizeof(cycles_ripemd));
    cudaMemcpyFromSymbol(&cycles_total, d_cycles_total, sizeof(cycles_total));
    cudaMemcpyFromSymbol(&count, d_count_ops, sizeof(count));
    
    // Calculate per-operation averages
    double avg_mul = (double)cycles_mul / (count * TEST_ITERATIONS);
    double avg_sqr = (double)cycles_sqr / (count * TEST_ITERATIONS);
    double avg_sub = (double)cycles_sub / (count * TEST_ITERATIONS);
    double avg_add = (double)cycles_add / (count * TEST_ITERATIONS);
    double avg_inv = (double)cycles_inv / (count * 10);  // Only 10 iterations
    double avg_sha = (double)cycles_sha / (count * TEST_ITERATIONS);
    double avg_ripemd = (double)cycles_ripemd / (count * TEST_ITERATIONS);
    
    printf("Cycles per Operation (averaged over %llu blocks):\n\n", count);
    printf("  ┌────────────────────┬────────────┬────────────┐\n");
    printf("  │ Operation          │ Cycles     │ Relative   │\n");
    printf("  ├────────────────────┼────────────┼────────────┤\n");
    printf("  │ fieldAdd           │ %10.1f │ %8.2fx  │\n", avg_add, avg_add/avg_add);
    printf("  │ fieldSub           │ %10.1f │ %8.2fx  │\n", avg_sub, avg_sub/avg_add);
    printf("  │ fieldMul           │ %10.1f │ %8.2fx  │\n", avg_mul, avg_mul/avg_add);
    printf("  │ fieldSqr           │ %10.1f │ %8.2fx  │\n", avg_sqr, avg_sqr/avg_add);
    printf("  │ fieldInv (Fermat)  │ %10.1f │ %8.2fx  │\n", avg_inv, avg_inv/avg_add);
    printf("  │ SHA256             │ %10.1f │ %8.2fx  │\n", avg_sha, avg_sha/avg_add);
    printf("  │ RIPEMD160          │ %10.1f │ %8.2fx  │\n", avg_ripemd, avg_ripemd/avg_add);
    printf("  └────────────────────┴────────────┴────────────┘\n\n");
    
    // Calculate cost breakdown for one hash160 search iteration
    // Per point: ~10 fieldMul + ~3 fieldSqr + ~8 fieldSub + 1 fieldInv (amortized) + 1 SHA256 + 1 RIPEMD160
    double cost_ecc = 10*avg_mul + 3*avg_sqr + 8*avg_sub + avg_inv/256;  // inv amortized over batch
    double cost_hash = avg_sha + avg_ripemd;
    double cost_total = cost_ecc + cost_hash;
    
    printf("Cost Breakdown (per hash160 check, with batch inversion):\n\n");
    printf("  ECC (point add):     %10.1f cycles (%.1f%%)\n", cost_ecc, 100*cost_ecc/cost_total);
    printf("  Hash (SHA+RIPEMD):   %10.1f cycles (%.1f%%)\n", cost_hash, 100*cost_hash/cost_total);
    printf("  ─────────────────────────────────────────────\n");
    printf("  TOTAL:               %10.1f cycles\n\n", cost_total);
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("PART 2: Full Pipeline Profiling (256 points per thread)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    unsigned long long *d_ecc, *d_hash;
    cudaMalloc(&d_ecc, sizeof(unsigned long long));
    cudaMalloc(&d_hash, sizeof(unsigned long long));
    cudaMemcpy(d_ecc, &zero, sizeof(zero), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash, &zero, sizeof(zero), cudaMemcpyHostToDevice);
    
    kernel_pipeline_profile<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_ecc, d_hash);
    cudaDeviceSynchronize();
    
    unsigned long long ecc_total, hash_total;
    cudaMemcpy(&ecc_total, d_ecc, sizeof(ecc_total), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hash_total, d_hash, sizeof(hash_total), cudaMemcpyDeviceToHost);
    
    double ecc_pct = 100.0 * ecc_total / (ecc_total + hash_total);
    double hash_pct = 100.0 * hash_total / (ecc_total + hash_total);
    
    printf("Pipeline Breakdown (realistic simulation):\n\n");
    printf("  ECC (with per-point fieldInv):  %.1f%%\n", ecc_pct);
    printf("  Hash (SHA256 + RIPEMD160):      %.1f%%\n\n", hash_pct);
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("OPTIMIZATION TARGETS (highest impact first):\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    if(avg_inv > avg_sha * 10) {
        printf("  🔥 #1 CRITICAL: fieldInv_Fermat (%.0f cycles)\n", avg_inv);
        printf("     → Solution: Batch inversion amortizes this over 256 points\n");
        printf("     → Already implemented, reduces effective cost to %.1f cycles/point\n\n", avg_inv/256);
    }
    
    if(avg_sha > avg_mul * 5) {
        printf("  🔥 #2 HIGH: SHA256 (%.0f cycles)\n", avg_sha);
        printf("     → 64 rounds of mixing, 32-bit operations\n");
        printf("     → Potential: Warp-level parallelism, lookup tables\n\n");
    }
    
    if(avg_ripemd > avg_mul * 3) {
        printf("  🔥 #3 MEDIUM: RIPEMD160 (%.0f cycles)\n", avg_ripemd);
        printf("     → 80 rounds, similar structure to SHA\n");
        printf("     → Potential: Combined SHA+RIPEMD optimization\n\n");
    }
    
    printf("  📊 #4 LOW: fieldMul (%.0f cycles) - already fast\n", avg_mul);
    printf("  📊 #5 LOW: fieldSqr (%.0f cycles) - already fast\n\n", avg_sqr);
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("THEORETICAL MAXIMUM THROUGHPUT:\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    // RTX 3090: ~1.7 GHz boost, 82 SMs, 128 CUDA cores per SM
    double gpu_freq_ghz = 1.7;
    double total_cores = 82 * 128;
    double cycles_per_hash = cost_total;
    double hashes_per_second = (gpu_freq_ghz * 1e9 * total_cores) / cycles_per_hash;
    
    printf("  GPU: RTX 3090 (%.1f GHz, %.0f CUDA cores)\n", gpu_freq_ghz, total_cores);
    printf("  Cycles per hash160: %.0f\n", cycles_per_hash);
    printf("  Theoretical max: %.2f GKeys/s\n\n", hashes_per_second / 1e9);
    
    printf("  Current achieved: ~2 GKeys/s\n");
    printf("  Efficiency: %.1f%%\n\n", 100 * 2e9 / hashes_per_second);
    
    cudaFree(d_ecc);
    cudaFree(d_hash);
    
    return 0;
}
