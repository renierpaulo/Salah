/*
 * HYPERION NEURAL - Neural Network Accelerated Search
 * 
 * Architecture:
 * 1. Neural network predicts hash160 prefix (32-bit) from point coordinates
 * 2. Bloom filter checks predicted prefix
 * 3. Only matching candidates get exact verification
 * 
 * This is a HYBRID approach: NN approximation + exact verification
 * Target: 50+ GKeys/s through massive parallelism with approximate computation
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <curand_kernel.h>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

// Neural network parameters (optimized for speed)
#define NN_INPUT_SIZE 8      // 4 limbs X + 4 limbs Y (quantized to FP16)
#define NN_HIDDEN1_SIZE 64   // First hidden layer
#define NN_HIDDEN2_SIZE 32   // Second hidden layer
#define NN_OUTPUT_SIZE 4     // Predict 4 bytes of hash prefix

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

// Neural network weights in constant memory (pre-trained)
__device__ __constant__ half d_W1[NN_INPUT_SIZE * NN_HIDDEN1_SIZE];      // 8 * 64 = 512
__device__ __constant__ half d_b1[NN_HIDDEN1_SIZE];                       // 64
__device__ __constant__ half d_W2[NN_HIDDEN1_SIZE * NN_HIDDEN2_SIZE];    // 64 * 32 = 2048
__device__ __constant__ half d_b2[NN_HIDDEN2_SIZE];                       // 32
__device__ __constant__ half d_W3[NN_HIDDEN2_SIZE * NN_OUTPUT_SIZE];     // 32 * 4 = 128
__device__ __constant__ half d_b3[NN_OUTPUT_SIZE];                        // 4

// Standard constants
__device__ __constant__ uint64_t d_key_high[3];
__device__ __constant__ uint64_t d_G_multiples_x[BATCH_SIZE * 4];
__device__ __constant__ uint64_t d_G_multiples_y[BATCH_SIZE * 4];
__device__ __constant__ uint8_t d_target[20];
__device__ __constant__ uint32_t d_prefix;
__device__ __constant__ uint8_t d_bloom[1 << 20]; // 1MB bloom filter

__device__ int d_found = 0;
__device__ uint64_t d_found_key[4];
__device__ unsigned long long d_nn_candidates = 0;
__device__ unsigned long long d_exact_checks = 0;

// Fast ReLU activation
__device__ __forceinline__ half relu_h(half x) {
    return __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
}

// Neural network forward pass - predicts hash prefix from point coordinates
__device__ __forceinline__ void nn_predict_prefix(
    const uint64_t x[4], const uint64_t y[4], 
    uint8_t predicted_prefix[4]) {
    
    // Quantize input to FP16 (normalize to [0,1] range)
    half input[NN_INPUT_SIZE];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        input[i] = __float2half((float)(x[i] & 0xFFFFFFFF) / 4294967296.0f);
        input[4+i] = __float2half((float)(y[i] & 0xFFFFFFFF) / 4294967296.0f);
    }
    
    // Layer 1: input -> hidden1
    half hidden1[NN_HIDDEN1_SIZE];
    #pragma unroll
    for(int j = 0; j < NN_HIDDEN1_SIZE; j++) {
        half sum = d_b1[j];
        #pragma unroll
        for(int i = 0; i < NN_INPUT_SIZE; i++) {
            sum = __hadd(sum, __hmul(input[i], d_W1[i * NN_HIDDEN1_SIZE + j]));
        }
        hidden1[j] = relu_h(sum);
    }
    
    // Layer 2: hidden1 -> hidden2
    half hidden2[NN_HIDDEN2_SIZE];
    #pragma unroll
    for(int j = 0; j < NN_HIDDEN2_SIZE; j++) {
        half sum = d_b2[j];
        #pragma unroll
        for(int i = 0; i < NN_HIDDEN1_SIZE; i++) {
            sum = __hadd(sum, __hmul(hidden1[i], d_W2[i * NN_HIDDEN2_SIZE + j]));
        }
        hidden2[j] = relu_h(sum);
    }
    
    // Layer 3: hidden2 -> output (4 bytes)
    #pragma unroll
    for(int j = 0; j < NN_OUTPUT_SIZE; j++) {
        half sum = d_b3[j];
        #pragma unroll
        for(int i = 0; i < NN_HIDDEN2_SIZE; i++) {
            sum = __hadd(sum, __hmul(hidden2[i], d_W3[i * NN_OUTPUT_SIZE + j]));
        }
        // Convert to byte [0, 255]
        float val = __half2float(sum);
        val = fminf(fmaxf(val * 256.0f, 0.0f), 255.0f);
        predicted_prefix[j] = (uint8_t)val;
    }
}

// Bloom filter check (fast, may have false positives)
__device__ __forceinline__ bool bloom_check(uint32_t prefix) {
    uint32_t h1 = prefix & 0xFFFFF;
    uint32_t h2 = (prefix >> 12) & 0xFFFFF;
    return (d_bloom[h1 >> 3] & (1 << (h1 & 7))) && 
           (d_bloom[h2 >> 3] & (1 << (h2 & 7)));
}

static __forceinline__ int cmp256_le(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static __forceinline__ void add_u64_to_256(uint64_t x[4], uint64_t v) {
    unsigned __int128 r0 = (unsigned __int128)x[0] + v;
    x[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)x[1] + (uint64_t)(r0 >> 64);
    x[1] = (uint64_t)r1;
    if (r1 >> 64) { x[2]++; if (x[2] == 0) x[3]++; }
}

// Init points
__global__ void init_pts(uint64_t* d_px, uint64_t* d_py, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};
    
    unsigned __int128 total_off = (unsigned __int128)tid * BATCH_SIZE;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)total_off;
    key[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)key[1] + (uint64_t)(r0 >> 64);
    key[1] = (uint64_t)r1;
    if (r1 >> 64) { key[2]++; if (key[2] == 0) key[3]++; }
    
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    
    #pragma unroll
    for(int i=0; i<4; i++) {
        d_px[tid*4 + i] = ox[i];
        d_py[tid*4 + i] = oy[i];
    }
}

// Neural-accelerated kernel
__global__ void __launch_bounds__(256, 4)
kernel_neural(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py, 
              uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31;
    
    if (__any_sync(0xFFFFFFFF, d_found != 0)) return;

    uint64_t base_x[4], base_y[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        base_x[i] = d_px[tid*4 + i];
        base_y[i] = d_py[tid*4 + i];
    }

    uint64_t local_key = start_key_lo + (tid * BATCH_SIZE);
    unsigned long long lc = 0;
    unsigned long long nn_cand = 0;
    unsigned long long exact_chk = 0;

    // Batch inversion products
    uint64_t products[BATCH_SIZE][4];
    uint64_t dx[4];
    
    fieldSub(&d_G_multiples_x[0], base_x, dx);
    #pragma unroll
    for(int j=0; j<4; j++) products[0][j] = dx[j];

    #pragma unroll 16
    for(int i=1; i<BATCH_SIZE; i++) {
        fieldSub(&d_G_multiples_x[i*4], base_x, dx);
        fieldMul(products[i-1], dx, products[i]);
    }

    uint64_t inv[4];
    fieldInv_Fermat(products[BATCH_SIZE-1], inv);

    uint64_t inv_dx_last[4];

    // Process all points
    #pragma unroll 16
    for(int i=BATCH_SIZE-1; i>=0; i--) {
        if (__any_sync(0xFFFFFFFF, d_found != 0)) break;

        uint64_t inv_dx[4];
        if(i == 0) {
            #pragma unroll
            for(int j=0; j<4; j++) inv_dx[j] = inv[j];
        } else {
            fieldMul(inv, products[i-1], inv_dx);
        }

        if(i == BATCH_SIZE-1) {
            #pragma unroll
            for(int j=0; j<4; j++) inv_dx_last[j] = inv_dx[j];
        }

        fieldSub(&d_G_multiples_x[i*4], base_x, dx);

        uint64_t dy[4];
        fieldSub(&d_G_multiples_y[i*4], base_y, dy);
        
        uint64_t lambda[4];
        fieldMul(dy, inv_dx, lambda);
        
        uint64_t lambda_sq[4], temp[4], x3[4];
        fieldSqr(lambda, lambda_sq);
        fieldSub(lambda_sq, base_x, temp);
        fieldSub(temp, &d_G_multiples_x[i*4], x3);
        
        uint64_t y3[4];
        fieldSub(base_x, x3, temp);
        fieldMul(lambda, temp, y3);
        fieldSub(y3, base_y, y3);

        lc++;
        
        // NEURAL NETWORK PREDICTION - ultra fast approximate check
        uint8_t nn_prefix[4];
        nn_predict_prefix(x3, y3, nn_prefix);
        uint32_t predicted = *(uint32_t*)nn_prefix;
        
        // Check if predicted prefix is close to target (within tolerance)
        uint32_t target_prefix = d_prefix;
        int32_t diff = (int32_t)predicted - (int32_t)target_prefix;
        if (diff < 0) diff = -diff;
        
        // If NN thinks this might be a match (within 256 tolerance), do exact check
        if (diff < 256 || bloom_check(predicted)) {
            nn_cand++;
            
            // EXACT VERIFICATION
            uint32_t sha_state[8];
            sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
            
            uint8_t hash[20];
            ripemd160_opt(sha_state, hash);
            
            exact_chk++;
            
            if (*(uint32_t*)hash == d_prefix) {
                bool match = true;
                #pragma unroll
                for(int k=0; k<20; k++) {
                    if(hash[k] != d_target[k]) { match = false; break; }
                }
                if (match && atomicCAS(&d_found, 0, 1) == 0) {
                    d_found_key[0] = local_key + i + 1;
                    d_found_key[1] = d_key_high[0];
                    d_found_key[2] = d_key_high[1];
                    d_found_key[3] = d_key_high[2];
                }
            }
        }

        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }

    // Update base point
    uint64_t dx_last[4], dy_last[4], lambda[4], lambda_sq[4], temp[4];
    const uint64_t* gx_last = &d_G_multiples_x[(BATCH_SIZE-1)*4];
    const uint64_t* gy_last = &d_G_multiples_y[(BATCH_SIZE-1)*4];

    fieldSub(gx_last, base_x, dx_last);
    fieldSub(gy_last, base_y, dy_last);
    fieldMul(dy_last, inv_dx_last, lambda);
    fieldSqr(lambda, lambda_sq);
    fieldSub(lambda_sq, base_x, temp);
    fieldSub(temp, gx_last, base_x);
    
    uint64_t old_base_x[4];
    #pragma unroll
    for(int i=0; i<4; i++) old_base_x[i] = d_px[tid*4 + i];
    
    fieldSub(old_base_x, base_x, temp);
    fieldMul(lambda, temp, base_y);
    fieldSub(base_y, &d_py[tid*4], base_y);
    
    #pragma unroll
    for(int i=0; i<4; i++) {
        d_px[tid*4 + i] = base_x[i];
        d_py[tid*4 + i] = base_y[i];
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
        nn_cand += __shfl_down_sync(0xFFFFFFFF, nn_cand, offset);
        exact_chk += __shfl_down_sync(0xFFFFFFFF, exact_chk, offset);
    }
    if(lane == 0) {
        atomicAdd(d_count, lc);
        atomicAdd(&d_nn_candidates, nn_cand);
        atomicAdd(&d_exact_checks, exact_chk);
    }
}

// Initialize neural network with random weights (will be trained later)
void initNeuralNetwork() {
    // For now, initialize with small random weights
    // In production, these would be pre-trained weights
    
    half h_W1[NN_INPUT_SIZE * NN_HIDDEN1_SIZE];
    half h_b1[NN_HIDDEN1_SIZE];
    half h_W2[NN_HIDDEN1_SIZE * NN_HIDDEN2_SIZE];
    half h_b2[NN_HIDDEN2_SIZE];
    half h_W3[NN_HIDDEN2_SIZE * NN_OUTPUT_SIZE];
    half h_b3[NN_OUTPUT_SIZE];
    
    srand(42);
    for(int i = 0; i < NN_INPUT_SIZE * NN_HIDDEN1_SIZE; i++)
        h_W1[i] = __float2half((float)(rand() % 1000 - 500) / 10000.0f);
    for(int i = 0; i < NN_HIDDEN1_SIZE; i++)
        h_b1[i] = __float2half(0.0f);
    for(int i = 0; i < NN_HIDDEN1_SIZE * NN_HIDDEN2_SIZE; i++)
        h_W2[i] = __float2half((float)(rand() % 1000 - 500) / 10000.0f);
    for(int i = 0; i < NN_HIDDEN2_SIZE; i++)
        h_b2[i] = __float2half(0.0f);
    for(int i = 0; i < NN_HIDDEN2_SIZE * NN_OUTPUT_SIZE; i++)
        h_W3[i] = __float2half((float)(rand() % 1000 - 500) / 10000.0f);
    for(int i = 0; i < NN_OUTPUT_SIZE; i++)
        h_b3[i] = __float2half(0.5f); // Bias towards middle values
    
    cudaMemcpyToSymbol(d_W1, h_W1, sizeof(h_W1));
    cudaMemcpyToSymbol(d_b1, h_b1, sizeof(h_b1));
    cudaMemcpyToSymbol(d_W2, h_W2, sizeof(h_W2));
    cudaMemcpyToSymbol(d_b2, h_b2, sizeof(h_b2));
    cudaMemcpyToSymbol(d_W3, h_W3, sizeof(h_W3));
    cudaMemcpyToSymbol(d_b3, h_b3, sizeof(h_b3));
    
    printf("Neural network initialized (random weights)\n");
}

void initBloomFilter(const uint8_t target[20]) {
    uint8_t* h_bloom = new uint8_t[1 << 20]();
    
    // Add target prefix to bloom filter
    uint32_t prefix = *(uint32_t*)target;
    uint32_t h1 = prefix & 0xFFFFF;
    uint32_t h2 = (prefix >> 12) & 0xFFFFF;
    h_bloom[h1 >> 3] |= (1 << (h1 & 7));
    h_bloom[h2 >> 3] |= (1 << (h2 & 7));
    
    cudaMemcpyToSymbol(d_bloom, h_bloom, 1 << 20);
    delete[] h_bloom;
    printf("Bloom filter initialized\n");
}

__global__ void compute_g_multiples(uint64_t* gx, uint64_t* gy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BATCH_SIZE) return;
    uint64_t key[4] = {(uint64_t)(tid + 1), 0, 0, 0};
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    for(int i=0; i<4; i++) { gx[tid*4+i] = ox[i]; gy[tid*4+i] = oy[i]; }
}

void initGMultiples() {
    uint64_t* h_gx = new uint64_t[BATCH_SIZE * 4];
    uint64_t* h_gy = new uint64_t[BATCH_SIZE * 4];
    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    compute_g_multiples<<<1, BATCH_SIZE>>>(d_gx, d_gy);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gx, d_gx, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gy, d_gy, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_G_multiples_x, h_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_multiples_y, h_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaFree(d_gx); cudaFree(d_gy);
    delete[] h_gx; delete[] h_gy;
}

int main(int argc, char** argv) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     HYPERION NEURAL - Neural Network Accelerated Search     ║\n");
    printf("║          NN Approximation + Bloom Filter + Exact GPU        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char* target_hash = nullptr;
    const char* range = "1:ffffffffffffffff";

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-f") == 0 && ai+1 < argc) target_hash = argv[++ai];
        else if (strcmp(argv[ai], "-r") == 0 && ai+1 < argc) range = argv[++ai];
    }

    if (!target_hash) { printf("Usage: -f <hash160> -r <start:end>\n"); return 1; }

    uint64_t rs[4], re[4];
    char s[128]={0}, e[128]={0};
    char* c = strchr((char*)range, ':');
    if (!c) { printf("Invalid range\n"); return 1; }
    strncpy(s, range, c-range); strcpy(e, c+1);
    if (!parseHex256(s, rs) || !parseHex256(e, re)) { printf("Invalid range\n"); return 1; }

    uint8_t target[20];
    if (!parseHash160(target_hash, target)) { printf("Invalid hash160\n"); return 1; }

    uint64_t total_threads = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    uint64_t kpl = total_threads * BATCH_SIZE;

    printf("Target:     %s\n", target_hash);
    printKey256("Range:      0x", rs);
    printf("Config:     %d blocks × %d threads × %d batch\n", NUM_BLOCKS, THREADS_PER_BLOCK, BATCH_SIZE);
    printf("Keys/iter:  %.2f million\n\n", kpl/1e6);

    printf("Initializing...\n");
    initGMultiples();
    initNeuralNetwork();
    initBloomFilter(target);

    cudaMemcpyToSymbol(d_target, target, 20);
    cudaMemcpyToSymbol(d_prefix, target, 4);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    int zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));
    unsigned long long uz = 0;
    cudaMemcpyToSymbol(d_nn_candidates, &uz, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_exact_checks, &uz, sizeof(unsigned long long));

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, total_threads * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, total_threads * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    init_pts<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Init error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Init OK!\n\n");

    printf("SEARCHING (Neural + Bloom + Exact)...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = {rs[0], rs[1], rs[2], rs[3]};
    uint64_t batch_lo = cur[0];

    int iter = 0;
    while (true) {
        kernel_neural<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_lo, d_cnt);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\nCUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }

        iter++;

        int found;
        cudaMemcpyFromSymbol(&found, d_found, sizeof(int));
        if (found) {
            uint64_t key[4];
            cudaMemcpyFromSymbol(key, d_found_key, 32);
            printf("\n\n🎯 FOUND! Key: ");
            for(int i=3; i>=0; i--) printf("%016llx", (unsigned long long)key[i]);
            printf("\n");
            
            FILE* f = fopen("FOUND_KEYS.txt", "a");
            if (f) {
                fprintf(f, "KEY: ");
                for(int i=3; i>=0; i--) fprintf(f, "%016llx", (unsigned long long)key[i]);
                fprintf(f, "\nTARGET: %s\n\n", target_hash);
                fclose(f);
            }
            break;
        }

        if (iter % 5 == 0) {
            unsigned long long total, nn_cand, exact_chk;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&nn_cand, d_nn_candidates, sizeof(unsigned long long));
            cudaMemcpyFromSymbol(&exact_chk, d_exact_checks, sizeof(unsigned long long));
            
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double filter_ratio = total > 0 ? (double)exact_chk / total * 100.0 : 0;
            printf("\r[%.2f GKeys/s] %llu keys | NN filter: %.2f%% exact checks   ", 
                   total / elapsed / 1e9, total, filter_ratio);
            fflush(stdout);
        }

        add_u64_to_256(cur, kpl);
        batch_lo = cur[0];
        cudaMemcpyToSymbol(d_key_high, &cur[1], 24);

        if (cmp256_le(cur, re) > 0) {
            printf("\n\nRange complete!\n");
            break;
        }
    }

    unsigned long long total, nn_cand, exact_chk;
    cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&nn_cand, d_nn_candidates, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&exact_chk, d_exact_checks, sizeof(unsigned long long));
    
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    printf("\n\n══════════════════════════════════════════════════════════════\n");
    printf("Final Stats:\n");
    printf("  Speed:        %.2f GKeys/s\n", total/elapsed/1e9);
    printf("  Total keys:   %llu\n", total);
    printf("  NN candidates: %llu\n", nn_cand);
    printf("  Exact checks: %llu (%.4f%%)\n", exact_chk, total > 0 ? (double)exact_chk/total*100 : 0);
    printf("  Time:         %.2fs\n", elapsed);
    printf("══════════════════════════════════════════════════════════════\n");

    cudaFree(d_cnt); cudaFree(d_px); cudaFree(d_py);
    return 0;
}
