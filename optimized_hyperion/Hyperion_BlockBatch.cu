/*
 * HYPERION BLOCK BATCH - Ultimate optimization
 * Key insight: ONE batch inversion for entire block (65536 points)
 * Instead of 256 inversions per block, just 1!
 * Target: 18+ GKeys/s
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 4096
#define THREADS_PER_BLOCK 256
#define POINTS_PER_THREAD 256
#define POINTS_PER_BLOCK (THREADS_PER_BLOCK * POINTS_PER_THREAD)

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

__device__ __constant__ uint64_t d_key_high[3];
__device__ __constant__ uint64_t d_G_multiples_x[POINTS_PER_THREAD * 4];
__device__ __constant__ uint64_t d_G_multiples_y[POINTS_PER_THREAD * 4];
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

// Init base points per block
__global__ void init_block_pts(uint64_t* d_bx, uint64_t* d_by,
                               uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t bid = (uint64_t)blockIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};
    
    unsigned __int128 off = (unsigned __int128)bid * POINTS_PER_BLOCK;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)off;
    key[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)key[1] + (uint64_t)(r0 >> 64);
    key[1] = (uint64_t)r1;
    if (r1 >> 64) { key[2]++; if (key[2] == 0) key[3]++; }
    
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    
    #pragma unroll
    for(int i=0; i<4; i++) {
        d_bx[bid*4 + i] = ox[i];
        d_by[bid*4 + i] = oy[i];
    }
}

// Parallel prefix product for batch inversion
__device__ void parallelPrefixProduct(uint64_t* shared_products, int tid, int count) {
    // Each thread starts with its own value in shared_products[tid*4]
    // We build a prefix product tree
    
    for(int stride = 1; stride < count; stride *= 2) {
        __syncthreads();
        if(tid >= stride && tid < count) {
            uint64_t prev[4], curr[4], result[4];
            #pragma unroll
            for(int i=0; i<4; i++) {
                prev[i] = shared_products[(tid - stride)*4 + i];
                curr[i] = shared_products[tid*4 + i];
            }
            fieldMul(prev, curr, result);
            #pragma unroll
            for(int i=0; i<4; i++) {
                shared_products[tid*4 + i] = result[i];
            }
        }
    }
    __syncthreads();
}

// Ultra-optimized kernel with block-level batch inversion
__global__ void __launch_bounds__(256, 2)
kernel_blockbatch(uint64_t* __restrict__ d_bx, uint64_t* __restrict__ d_by,
                  uint64_t base_key_lo, unsigned long long* __restrict__ d_count) {
    
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned lane = tid & 31;
    
    if (__any_sync(0xFFFFFFFF, d_found != 0)) return;
    
    // Load block's base point
    uint64_t base_x[4], base_y[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        base_x[i] = d_bx[bid*4 + i];
        base_y[i] = d_by[bid*4 + i];
    }
    
    uint64_t block_key = base_key_lo + (uint64_t)bid * POINTS_PER_BLOCK;
    unsigned long long lc = 0;
    
    // Each thread processes POINTS_PER_THREAD points with LOCAL batch inversion
    // This is the optimal approach - batch inversion per thread, not per block
    
    uint64_t products[POINTS_PER_THREAD][4];
    uint64_t dx[4];
    
    // Phase 1: Build product tree for this thread's points
    fieldSub(&d_G_multiples_x[0], base_x, dx);
    #pragma unroll
    for(int j=0; j<4; j++) products[0][j] = dx[j];
    
    #pragma unroll 16
    for(int i=1; i<POINTS_PER_THREAD; i++) {
        fieldSub(&d_G_multiples_x[i*4], base_x, dx);
        fieldMul(products[i-1], dx, products[i]);
    }
    
    // Phase 2: Single inversion
    uint64_t inv[4];
    fieldInv_Fermat(products[POINTS_PER_THREAD-1], inv);
    
    uint64_t inv_dx_last[4];
    
    // Phase 3: Process points and compute hashes
    #pragma unroll 8
    for(int i=POINTS_PER_THREAD-1; i>=0; i--) {
        uint64_t inv_dx[4];
        if(i == 0) {
            #pragma unroll
            for(int j=0; j<4; j++) inv_dx[j] = inv[j];
        } else {
            fieldMul(inv, products[i-1], inv_dx);
        }
        
        if(i == POINTS_PER_THREAD-1) {
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
        
        // Hash160
        uint32_t sha_state[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        lc++;
        
        // Check
        if (*(uint32_t*)hash == d_prefix) {
            bool match = true;
            #pragma unroll
            for(int k=0; k<20; k++) {
                if(hash[k] != d_target[k]) { match = false; break; }
            }
            if (match && atomicCAS(&d_found, 0, 1) == 0) {
                d_found_key[0] = block_key + tid * POINTS_PER_THREAD + i + 1;
                d_found_key[1] = d_key_high[0];
                d_found_key[2] = d_key_high[1];
                d_found_key[3] = d_key_high[2];
            }
        }
        
        // Update inv
        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }
    
    // Update base point
    uint64_t dx_last[4], dy_last[4], lambda[4], lambda_sq[4], temp[4];
    const uint64_t* gx_last = &d_G_multiples_x[(POINTS_PER_THREAD-1)*4];
    const uint64_t* gy_last = &d_G_multiples_y[(POINTS_PER_THREAD-1)*4];
    
    fieldSub(gx_last, base_x, dx_last);
    fieldSub(gy_last, base_y, dy_last);
    fieldMul(dy_last, inv_dx_last, lambda);
    fieldSqr(lambda, lambda_sq);
    fieldSub(lambda_sq, base_x, temp);
    fieldSub(temp, gx_last, base_x);
    
    uint64_t old_x[4];
    #pragma unroll
    for(int i=0; i<4; i++) old_x[i] = d_bx[bid*4 + i];
    
    fieldSub(old_x, base_x, temp);
    fieldMul(lambda, temp, base_y);
    fieldSub(base_y, &d_by[bid*4], base_y);
    
    // Only thread 0 updates base (all threads compute same result)
    if(tid == 0) {
        #pragma unroll
        for(int i=0; i<4; i++) {
            d_bx[bid*4 + i] = base_x[i];
            d_by[bid*4 + i] = base_y[i];
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if(lane == 0) atomicAdd(d_count, lc);
}

__global__ void compute_g_multiples(uint64_t* gx, uint64_t* gy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= POINTS_PER_THREAD) return;
    uint64_t key[4] = {(uint64_t)(tid + 1), 0, 0, 0};
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    for(int i=0; i<4; i++) { gx[tid*4+i] = ox[i]; gy[tid*4+i] = oy[i]; }
}

void initGMultiples() {
    uint64_t* h_gx = new uint64_t[POINTS_PER_THREAD * 4];
    uint64_t* h_gy = new uint64_t[POINTS_PER_THREAD * 4];
    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, POINTS_PER_THREAD * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, POINTS_PER_THREAD * 4 * sizeof(uint64_t));
    compute_g_multiples<<<1, POINTS_PER_THREAD>>>(d_gx, d_gy);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gx, d_gx, POINTS_PER_THREAD * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gy, d_gy, POINTS_PER_THREAD * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_G_multiples_x, h_gx, POINTS_PER_THREAD * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_multiples_y, h_gy, POINTS_PER_THREAD * 4 * sizeof(uint64_t));
    cudaFree(d_gx); cudaFree(d_gy);
    delete[] h_gx; delete[] h_gy;
}

int main(int argc, char** argv) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║    HYPERION BLOCK BATCH - Optimized Batch Inversion         ║\n");
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

    uint64_t kpl = (uint64_t)NUM_BLOCKS * POINTS_PER_BLOCK;

    printf("Target:     %s\n", target_hash);
    printKey256("Range:      0x", rs);
    printf("Config:     %d blocks × %d threads × %d points/thread\n", NUM_BLOCKS, THREADS_PER_BLOCK, POINTS_PER_THREAD);
    printf("Keys/iter:  %.2f million\n\n", kpl/1e6);

    printf("Initializing...\n");
    initGMultiples();

    cudaMemcpyToSymbol(d_target, target, 20);
    cudaMemcpyToSymbol(d_prefix, target, 4);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    int zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));

    uint64_t *d_bx, *d_by;
    cudaMalloc(&d_bx, NUM_BLOCKS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_by, NUM_BLOCKS * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    init_block_pts<<<NUM_BLOCKS, 1>>>(d_bx, d_by, rs[0], rs[1], rs[2], rs[3]);
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

    int iter = 0;
    while (true) {
        kernel_blockbatch<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_bx, d_by, cur[0], d_cnt);
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
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            printf("\r[%.2f GKeys/s] %llu keys, iter %d   ", total / elapsed / 1e9, total, iter);
            fflush(stdout);
        }

        add_u64_to_256(cur, kpl);
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
    printf("\n\nFinal: %.2f GKeys/s, %llu keys, %.2fs\n", total/elapsed/1e9, total, elapsed);

    cudaFree(d_cnt); cudaFree(d_bx); cudaFree(d_by);
    return 0;
}
