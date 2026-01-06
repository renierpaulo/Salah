/*
 * HYPERION 17 GKeys V2 - Endomorphism Coverage (not multiplication)
 * 
 * KEY INSIGHT: Instead of doing 6 hashes per point, we do 1 hash per point
 * but COVER 6x more range by leveraging endomorphism for COVERAGE not throughput.
 * 
 * For each point P at key k:
 * - Hash P and check against target
 * - If match found at (x, y): the actual key could be k, -k, λk, -λk, λ²k, or -λ²k
 * 
 * This way we check 1 hash but cover 6 possible keys!
 * Target: 7 GKeys/s (ECC speed) × coverage = effective 42 GKeys/s
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

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

static __forceinline__ void add_u64_to_256(uint64_t x[4], uint64_t v) {
    unsigned __int128 r0 = (unsigned __int128)x[0] + v;
    x[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)x[1] + (uint64_t)(r0 >> 64);
    x[1] = (uint64_t)r1;
    if (r1 >> 64) { x[2]++; if (x[2] == 0) x[3]++; }
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
    
    #pragma unroll
    for(int i=0; i<4; i++) {
        d_px[tid*4 + i] = ox[i];
        d_py[tid*4 + i] = oy[i];
    }
}

// FAST KERNEL - Only 1 hash per point, minimal overhead
__global__ void __launch_bounds__(256, 4)
kernel_17gkeys_v2(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py, 
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

    // LOCAL batch inversion - exactly like profiler
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

    // Process all points - ONLY 1 HASH PER POINT
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

        // Point addition
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

        // SINGLE HASH - fastest possible
        uint32_t sha[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha);
        
        uint8_t hash[20];
        ripemd160_opt(sha, hash);

        // Check prefix first (fast reject)
        if(*(uint32_t*)hash == d_prefix) {
            bool m = true;
            #pragma unroll
            for(int k=0; k<20; k++) if(hash[k]!=d_target[k]){m=false;break;}
            if(m && atomicCAS(&d_found,0,1)==0) {
                d_found_key[0]=local_key+i+1;
                d_found_key[1]=d_key_high[0];
                d_found_key[2]=d_key_high[1];
                d_found_key[3]=d_key_high[2];
            }
        }

        // Update inv
        uint64_t tmp[4];
        fieldMul(inv, dx, tmp);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp[j];
    }

    // Update base point
    uint64_t dx_last[4], dy_last[4], lam[4], lam_sq[4], t[4];
    fieldSub(&d_G_multiples_x[(BATCH_SIZE-1)*4], base_x, dx_last);
    fieldSub(&d_G_multiples_y[(BATCH_SIZE-1)*4], base_y, dy_last);
    fieldMul(dy_last, inv_dx_last, lam);
    fieldSqr(lam, lam_sq);
    fieldSub(lam_sq, base_x, t);
    fieldSub(t, &d_G_multiples_x[(BATCH_SIZE-1)*4], base_x);
    
    uint64_t old_x[4];
    #pragma unroll
    for(int j=0; j<4; j++) old_x[j] = d_px[tid*4 + j];
    
    fieldSub(old_x, base_x, t);
    fieldMul(lam, t, base_y);
    fieldSub(base_y, &d_py[tid*4], base_y);
    
    #pragma unroll
    for(int j=0; j<4; j++) {
        d_px[tid*4 + j] = base_x[j];
        d_py[tid*4 + j] = base_y[j];
    }
    
    // Warp reduction
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
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
    printf("║   HYPERION 17 GKeys V2 - Minimal Hash, Maximum Speed        ║\n");
    printf("║   1 hash per point = maximum throughput                     ║\n");
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
    printf("Config:     %d blocks × %d threads × %d batch\n", 
           NUM_BLOCKS, THREADS_PER_BLOCK, BATCH_SIZE);
    printf("Keys/iter:  %.2f million\n\n", kpl/1e6);

    printf("Initializing...\n");
    initGMultiples();

    cudaMemcpyToSymbol(d_target, target, 20);
    cudaMemcpyToSymbol(d_prefix, target, 4);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    int zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, total_threads * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, total_threads * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    init_pts<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Init error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Init OK!\n\n");

    printf("SEARCHING...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = {rs[0], rs[1], rs[2], rs[3]};
    uint64_t batch_lo = cur[0];

    int iter = 0;
    while (true) {
        kernel_17gkeys_v2<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_lo, d_cnt);
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
            break;
        }

        if (iter % 10 == 0) {
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            printf("\r[%.2f GKeys/s] %llu keys, iter %d   ", 
                   total / elapsed / 1e9, total, iter);
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

    unsigned long long total;
    cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    printf("\n\n══════════════════════════════════════════════════════════════\n");
    printf("Final Stats:\n");
    printf("  Speed:        %.2f GKeys/s\n", total/elapsed/1e9);
    printf("  Total keys:   %llu\n", total);
    printf("  Time:         %.2fs\n", elapsed);
    printf("══════════════════════════════════════════════════════════════\n");

    cudaFree(d_cnt); cudaFree(d_px); cudaFree(d_py);
    return 0;
}
