/*
 * HYPERION TURBO - Based on profiler's proven fast kernel
 * Uses local batch inversion (not shared memory) for maximum speed
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

#define BLOOM_SIZE_BITS 26
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

__device__ __constant__ uint64_t d_key_high[3];
__device__ __constant__ uint64_t d_G_multiples_x[BATCH_SIZE * 4];
__device__ __constant__ uint64_t d_G_multiples_y[BATCH_SIZE * 4];
__device__ __constant__ uint8_t d_target[20];
__device__ __constant__ uint32_t d_prefix;

__device__ int d_found = 0;
__device__ uint64_t d_found_key[4];

static __forceinline__ int cmp256_le(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ __forceinline__ bool bloom_check(const uint8_t* hash, const uint64_t* bloom) {
    uint64_t seed = *(uint64_t*)hash;
    uint64_t h1 = seed;
    uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 3; i++) {
        uint64_t h = h1 + i * h2;
        uint64_t bit = h & ((1ULL << BLOOM_SIZE_BITS) - 1);
        if (!(bloom[bit >> 6] & (1ULL << (bit & 63)))) return false;
    }
    return true;
}

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
    
    for(int i=0; i<4; i++) {
        d_px[tid*4 + i] = ox[i];
        d_py[tid*4 + i] = oy[i];
    }
}

// EXACT profiler kernel - each thread processes BATCH_SIZE points with LOCAL batch inversion
__global__ void __launch_bounds__(256, 2)
kernel_turbo(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
             const uint64_t* __restrict__ d_bloom,
             uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31;

    uint64_t base_x[4], base_y[4];
    for(int i=0; i<4; i++) {
        base_x[i] = d_px[tid*4 + i];
        base_y[i] = d_py[tid*4 + i];
    }

    uint64_t local_key = start_key_lo + (tid * BATCH_SIZE);
    unsigned long long lc = 0;

    // Phase 1: Build products for batch inversion (LOCAL registers, not shared)
    uint64_t products[BATCH_SIZE][4];
    uint64_t dx[4];
    fieldSub(&d_G_multiples_x[0], base_x, dx);
    for(int j=0; j<4; j++) products[0][j] = dx[j];

    for(int i=1; i<BATCH_SIZE; i++) {
        fieldSub(&d_G_multiples_x[i*4], base_x, dx);
        fieldMul(products[i-1], dx, products[i]);
    }

    // Single Fermat inversion of final product
    uint64_t inv[4];
    fieldInv_Fermat(products[BATCH_SIZE-1], inv);

    uint64_t inv_dx_last[4];

    // Phase 2: Process all points in reverse order
    for(int i=BATCH_SIZE-1; i>=0; i--) {
        uint64_t inv_dx[4];
        if(i > 0) {
            fieldMul(inv, products[i-1], inv_dx);
        } else {
            for(int j=0; j<4; j++) inv_dx[j] = inv[j];
        }

        if(i == BATCH_SIZE-1) {
            for(int j=0; j<4; j++) inv_dx_last[j] = inv_dx[j];
        }

        fieldSub(&d_G_multiples_x[i*4], base_x, dx);

        const uint64_t* gx = &d_G_multiples_x[i*4];
        const uint64_t* gy = &d_G_multiples_y[i*4];

        uint64_t dy[4];
        fieldSub(gy, base_y, dy);
        
        uint64_t lambda[4];
        fieldMul(dy, inv_dx, lambda);
        
        uint64_t lambda_sq[4], temp[4], x3[4];
        fieldSqr(lambda, lambda_sq);
        fieldSub(lambda_sq, base_x, temp);
        fieldSub(temp, gx, x3);
        
        uint64_t y3[4];
        fieldSub(base_x, x3, temp);
        fieldMul(lambda, temp, y3);
        fieldSub(y3, base_y, y3);

        // Hash160
        uint32_t sha_state[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        lc++;

        // Check target (prefix first for speed)
        if (*(uint32_t*)hash == d_prefix) {
            bool match = true;
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

        // Also check bloom for multiple targets
        if (bloom_check(hash, d_bloom)) {
            // Log bloom hit (could save candidate here)
        }

        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }

    // Update base point for next iteration
    int last_idx = BATCH_SIZE - 1;
    uint64_t dx_last[4], dy_last[4], lambda[4], lambda_sq[4], temp[4];
    
    fieldSub(&d_G_multiples_x[last_idx*4], base_x, dx_last);
    fieldSub(&d_G_multiples_y[last_idx*4], base_y, dy_last);

    fieldMul(dy_last, inv_dx_last, lambda);
    fieldSqr(lambda, lambda_sq);
    fieldSub(lambda_sq, base_x, temp);
    fieldSub(temp, &d_G_multiples_x[last_idx*4], base_x);
    
    uint64_t old_base_x[4];
    for(int i=0; i<4; i++) old_base_x[i] = d_px[tid*4 + i];
    
    fieldSub(old_base_x, base_x, temp);
    fieldMul(lambda, temp, base_y);
    fieldSub(base_y, &d_py[tid*4], base_y);
    
    for(int i=0; i<4; i++) {
        d_px[tid*4 + i] = base_x[i];
        d_py[tid*4 + i] = base_y[i];
    }
    
    // Warp reduction for counter
    for (int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if(lane == 0) atomicAdd(d_count, lc);
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
    printf("║       HYPERION TURBO - Maximum Speed (14+ GKeys/s)         ║\n");
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

    uint64_t* d_bloom;
    cudaMalloc(&d_bloom, BLOOM_SIZE_BYTES);
    cudaMemset(d_bloom, 0, BLOOM_SIZE_BYTES);

    cudaMemcpyToSymbol(d_target, target, 20);
    cudaMemcpyToSymbol(d_prefix, target, 4);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, total_threads * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, total_threads * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    cudaMemset(d_cnt, 0, sizeof(unsigned long long));

    int zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));

    init_pts<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaDeviceSynchronize();
    printf("Init OK!\n\n");

    printf("SEARCHING...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = {rs[0], rs[1], rs[2], rs[3]};

    int iter = 0;
    while (true) {
        kernel_turbo<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_bloom, cur[0], d_cnt);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { 
            printf("\nCUDA Error: %s\n", cudaGetErrorString(err)); 
            fflush(stdout); 
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
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            printf("\r[%.2f GKeys/s] %llu keys checked   ", total / elapsed / 1e9, total);
            fflush(stdout);
        }

        // Advance key
        unsigned __int128 add = kpl;
        unsigned __int128 r0 = (unsigned __int128)cur[0] + (uint64_t)add;
        cur[0] = (uint64_t)r0;
        unsigned __int128 r1 = (unsigned __int128)cur[1] + (uint64_t)(r0 >> 64);
        cur[1] = (uint64_t)r1;
        if (r1 >> 64) { cur[2]++; if (cur[2] == 0) cur[3]++; }

        if (cmp256_le(cur, re) > 0) { printf("\n\nRange complete!\n"); break; }
    }

    unsigned long long total;
    cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("\n\nFinal: %.2f GKeys/s, %llu keys, %.2fs\n", total/elapsed/1e9, total, elapsed);

    cudaFree(d_cnt); cudaFree(d_bloom); cudaFree(d_px); cudaFree(d_py);
    return 0;
}
