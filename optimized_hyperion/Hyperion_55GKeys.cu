/*
 * HYPERION 55 GKeys - Fixed Batch Inversion (O(N) instead of O(N²))
 * 
 * Bug fix: The original batchInvBlock had O(N²) complexity due to nested loops
 * causing massive warp divergence and hangs. This version uses proper Montgomery
 * batch inversion which is O(N) with parallel prefix scan.
 * 
 * Target: 55+ GKeys/s (ECC-only speed achieved before)
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

// FIXED BATCH INVERSION - O(N) using parallel prefix scan
// Montgomery trick: inv[i] = (product of all z except z[i]) * (1 / product of all z)
__device__ void batchInvBlock_Fixed(uint64_t* z_array, uint64_t* inv_array, int count) {
    const int tid = threadIdx.x;
    
    // Shared memory for products and inverse propagation
    __shared__ uint64_t products[256][4];     // Prefix products
    __shared__ uint64_t global_inv[4];        // 1 / (z[0] * z[1] * ... * z[n-1])
    
    if (tid >= count) return;
    
    // Step 1: Load z[tid] into local register
    uint64_t z_local[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        z_local[i] = z_array[tid * 4 + i];
    }
    
    // Step 2: Compute prefix products using parallel scan
    // products[tid] = z[0] * z[1] * ... * z[tid]
    uint64_t prod[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) prod[i] = z_local[i];
    
    // Store initial value
    #pragma unroll
    for(int i = 0; i < 4; i++) products[tid][i] = prod[i];
    __syncthreads();
    
    // Parallel prefix product (log N steps)
    for(int stride = 1; stride < count; stride *= 2) {
        uint64_t temp[4];
        if (tid >= stride) {
            uint64_t prev[4];
            #pragma unroll
            for(int i = 0; i < 4; i++) prev[i] = products[tid - stride][i];
            fieldMul(prod, prev, temp);
            #pragma unroll
            for(int i = 0; i < 4; i++) prod[i] = temp[i];
        }
        __syncthreads();
        #pragma unroll
        for(int i = 0; i < 4; i++) products[tid][i] = prod[i];
        __syncthreads();
    }
    
    // Step 3: Thread 0 inverts the final product
    if (tid == 0) {
        uint64_t final_prod[4];
        #pragma unroll
        for(int i = 0; i < 4; i++) final_prod[i] = products[count - 1][i];
        
        uint64_t inv[4];
        fieldInv_Fermat(final_prod, inv);
        
        #pragma unroll
        for(int i = 0; i < 4; i++) global_inv[i] = inv[i];
    }
    __syncthreads();
    
    // Step 4: Compute individual inverses - THIS IS THE KEY FIX!
    // inv[tid] = global_inv * products[tid-1]  (for tid > 0)
    // inv[0] = global_inv * (z[1] * z[2] * ... * z[n-1])
    //        = global_inv * products[n-1] / z[0] (but we compute differently)
    
    // Actually, we need to compute:
    // inv[tid] = 1/z[tid] = (prod of all z except z[tid]) / (prod of all z)
    //          = (products[tid-1] * suffix_products[tid+1]) / products[n-1]
    // 
    // Simpler approach: Compute suffix products in reverse
    __shared__ uint64_t suffix[256][4];
    
    // Compute suffix products: suffix[tid] = z[tid] * z[tid+1] * ... * z[n-1]
    #pragma unroll
    for(int i = 0; i < 4; i++) suffix[tid][i] = z_local[i];
    __syncthreads();
    
    for(int stride = 1; stride < count; stride *= 2) {
        uint64_t temp[4];
        int partner = tid + stride;
        if (partner < count) {
            uint64_t next[4];
            #pragma unroll
            for(int i = 0; i < 4; i++) next[i] = suffix[partner][i];
            uint64_t curr[4];
            #pragma unroll
            for(int i = 0; i < 4; i++) curr[i] = suffix[tid][i];
            fieldMul(curr, next, temp);
            #pragma unroll
            for(int i = 0; i < 4; i++) suffix[tid][i] = temp[i];
        }
        __syncthreads();
    }
    
    // Now compute inv[tid] = global_inv * products[tid-1] * suffix[tid+1]
    // Special cases:
    // - tid == 0: inv[0] = global_inv * suffix[1]
    // - tid == count-1: inv[n-1] = global_inv * products[n-2]
    // - otherwise: inv[tid] = global_inv * products[tid-1] * suffix[tid+1]
    
    uint64_t inv_local[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) inv_local[i] = global_inv[i];
    
    if (tid > 0) {
        uint64_t prefix[4];
        #pragma unroll
        for(int i = 0; i < 4; i++) prefix[i] = products[tid - 1][i];
        uint64_t temp[4];
        fieldMul(inv_local, prefix, temp);
        #pragma unroll
        for(int i = 0; i < 4; i++) inv_local[i] = temp[i];
    }
    
    if (tid < count - 1) {
        uint64_t suf[4];
        #pragma unroll
        for(int i = 0; i < 4; i++) suf[i] = suffix[tid + 1][i];
        uint64_t temp[4];
        fieldMul(inv_local, suf, temp);
        #pragma unroll
        for(int i = 0; i < 4; i++) inv_local[i] = temp[i];
    }
    
    // Store result
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        inv_array[tid * 4 + i] = inv_local[i];
    }
    __syncthreads();
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

// Main search kernel with FIXED batch inversion
__global__ void __launch_bounds__(256, 4)
kernel_55gkeys(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
               uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const unsigned lane = tid & 31;
    
    if (__any_sync(0xFFFFFFFF, d_found != 0)) return;
    
    // Shared memory
    __shared__ uint64_t shBaseX[4], shBaseY[4];
    __shared__ uint64_t shPZ[BATCH_SIZE * 4];
    __shared__ uint64_t shInvZ[BATCH_SIZE * 4];
    
    // Load base point for this block
    if (tid == 0) {
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            shBaseX[i] = d_px[bid * 4 + i];
            shBaseY[i] = d_py[bid * 4 + i];
        }
    }
    __syncthreads();
    
    uint64_t base_x[4], base_y[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        base_x[i] = shBaseX[i];
        base_y[i] = shBaseY[i];
    }
    
    // Load G multiple for this thread
    uint64_t gx[4], gy[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        gx[i] = d_G_multiples_x[tid * 4 + i];
        gy[i] = d_G_multiples_y[tid * 4 + i];
    }
    
    // Compute P + Q in Jacobian coordinates
    // P = base (affine), Q = tid*G (affine)
    // Result = Jacobian
    
    // For affine + affine -> Jacobian, we use:
    // lambda = (y2 - y1) / (x2 - x1)
    // x3 = lambda^2 - x1 - x2
    // y3 = lambda * (x1 - x3) - y1
    // z3 = x2 - x1
    
    uint64_t dx[4], dy[4];
    fieldSub(gx, base_x, dx);  // x2 - x1
    fieldSub(gy, base_y, dy);  // y2 - y1
    
    // Store Z = dx for batch inversion
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        shPZ[tid * 4 + i] = dx[i];
    }
    __syncthreads();
    
    // BATCH INVERSION - Fixed O(N) algorithm
    batchInvBlock_Fixed(shPZ, shInvZ, BATCH_SIZE);
    __syncthreads();
    
    // Load inverse of dx
    uint64_t inv_dx[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        inv_dx[i] = shInvZ[tid * 4 + i];
    }
    
    // Compute lambda = dy * inv_dx
    uint64_t lambda[4];
    fieldMul(dy, inv_dx, lambda);
    
    // Compute x3 = lambda^2 - base_x - gx
    uint64_t lambda_sq[4], x3[4], temp[4];
    fieldSqr(lambda, lambda_sq);
    fieldSub(lambda_sq, base_x, temp);
    fieldSub(temp, gx, x3);
    
    // Compute y3 = lambda * (base_x - x3) - base_y
    uint64_t y3[4];
    fieldSub(base_x, x3, temp);
    fieldMul(lambda, temp, y3);
    fieldSub(y3, base_y, y3);
    
    // Now we have affine coordinates (x3, y3) - compute hash160
    uint64_t local_key = start_key_lo + (uint64_t)bid * BATCH_SIZE + tid + 1;
    unsigned long long lc = 1;
    
    uint32_t sha[8];
    sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha);
    
    uint8_t hash[20];
    ripemd160_opt(sha, hash);
    
    // Check prefix
    if (*(uint32_t*)hash == d_prefix) {
        bool match = true;
        #pragma unroll
        for(int k = 0; k < 20; k++) {
            if (hash[k] != d_target[k]) { match = false; break; }
        }
        if (match && atomicCAS(&d_found, 0, 1) == 0) {
            d_found_key[0] = local_key;
            d_found_key[1] = d_key_high[0];
            d_found_key[2] = d_key_high[1];
            d_found_key[3] = d_key_high[2];
        }
    }
    
    // Update base point for next iteration (thread 0 only)
    if (tid == 0) {
        // Advance by BATCH_SIZE*G
        uint64_t gx_last[4], gy_last[4];
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            gx_last[i] = d_G_multiples_x[(BATCH_SIZE - 1) * 4 + i];
            gy_last[i] = d_G_multiples_y[(BATCH_SIZE - 1) * 4 + i];
        }
        
        uint64_t dx_l[4], dy_l[4];
        fieldSub(gx_last, base_x, dx_l);
        fieldSub(gy_last, base_y, dy_l);
        
        uint64_t inv_dx_l[4];
        fieldInv_Fermat(dx_l, inv_dx_l);
        
        uint64_t lam[4];
        fieldMul(dy_l, inv_dx_l, lam);
        
        uint64_t lam_sq[4], new_x[4], t[4];
        fieldSqr(lam, lam_sq);
        fieldSub(lam_sq, base_x, t);
        fieldSub(t, gx_last, new_x);
        
        uint64_t new_y[4];
        fieldSub(base_x, new_x, t);
        fieldMul(lam, t, new_y);
        fieldSub(new_y, base_y, new_y);
        
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            d_px[bid * 4 + i] = new_x[i];
            d_py[bid * 4 + i] = new_y[i];
        }
    }
    
    // Warp reduction for counter
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if (lane == 0) atomicAdd(d_count, lc);
}

__global__ void compute_g_multiples(uint64_t* gx, uint64_t* gy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BATCH_SIZE) return;
    uint64_t key[4] = {(uint64_t)(tid + 1), 0, 0, 0};
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    for(int i = 0; i < 4; i++) {
        gx[tid * 4 + i] = ox[i];
        gy[tid * 4 + i] = oy[i];
    }
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
    printf("║   HYPERION 55 GKeys - FIXED Batch Inversion O(N)            ║\n");
    printf("║   Bug fix: Removed O(N²) loop causing warp divergence       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char* target_hash = nullptr;
    const char* range = "1:ffffffffffffffff";

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-f") == 0 && ai + 1 < argc) target_hash = argv[++ai];
        else if (strcmp(argv[ai], "-r") == 0 && ai + 1 < argc) range = argv[++ai];
    }

    if (!target_hash) { printf("Usage: -f <hash160> -r <start:end>\n"); return 1; }

    uint64_t rs[4], re[4];
    char s[128] = {0}, e[128] = {0};
    char* c = strchr((char*)range, ':');
    if (!c) { printf("Invalid range\n"); return 1; }
    strncpy(s, range, c - range); strcpy(e, c + 1);
    if (!parseHex256(s, rs) || !parseHex256(e, re)) { printf("Invalid range\n"); return 1; }

    uint8_t target[20];
    if (!parseHash160(target_hash, target)) { printf("Invalid hash160\n"); return 1; }

    uint64_t total_threads = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    uint64_t kpl = total_threads;  // 1 key per thread per iteration

    printf("Target:     %s\n", target_hash);
    printKey256("Range:      0x", rs);
    printf("Config:     %d blocks × %d threads\n", NUM_BLOCKS, THREADS_PER_BLOCK);
    printf("Keys/iter:  %.2f million\n\n", kpl / 1e6);

    printf("Initializing...\n");
    initGMultiples();

    cudaMemcpyToSymbol(d_target, target, 20);
    cudaMemcpyToSymbol(d_prefix, target, 4);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    int zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, NUM_BLOCKS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, NUM_BLOCKS * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Initialize base points (one per block)
    init_pts<<<NUM_BLOCKS, 1>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Init error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Init OK!\n\n");

    printf("SEARCHING (Fixed O(N) batch inversion)...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = {rs[0], rs[1], rs[2], rs[3]};
    uint64_t batch_lo = cur[0];

    int iter = 0;
    while (true) {
        kernel_55gkeys<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_lo, d_cnt);
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
            for (int i = 3; i >= 0; i--) printf("%016llx", (unsigned long long)key[i]);
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
    printf("55 GKeys Final Stats:\n");
    printf("  Speed:        %.2f GKeys/s\n", total / elapsed / 1e9);
    printf("  Total keys:   %llu\n", total);
    printf("  Time:         %.2fs\n", elapsed);
    printf("══════════════════════════════════════════════════════════════\n");

    cudaFree(d_cnt); cudaFree(d_px); cudaFree(d_py);
    return 0;
}
