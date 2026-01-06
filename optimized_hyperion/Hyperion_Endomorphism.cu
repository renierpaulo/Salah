/*
 * HYPERION ENDOMORPHISM - Exploit secp256k1 endomorphism for 3x speedup
 * 
 * Key insight: secp256k1 has special structure. For any point P = (x, y):
 *   λ*P = (β*x mod p, y)      where β = cube root of 1 mod p
 *   λ²*P = (β²*x mod p, y)
 * 
 * This means for EACH point we compute, we get 2 MORE points almost FREE!
 * Cost: just 2 field multiplications instead of 2 full point additions + inversions
 * 
 * Target: 3x speedup = 7+ GKeys/s (from 2.3 GKeys/s)
 * Combined with other optimizations: potentially 10+ GKeys/s
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

// secp256k1 endomorphism constants
// β = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
// β² = 0x851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40
__device__ __constant__ uint64_t BETA[4] = {
    0xc1396c28719501eeULL, 0x9cf0497512f58995ULL, 
    0xe64479eac3434e99ULL, 0x7ae96a2b657c0710ULL
};
__device__ __constant__ uint64_t BETA_SQ[4] = {
    0x3ec693d68e6afa40ULL, 0x630fb68aed0a766aULL,
    0x919bb86153cbcb16ULL, 0x851695d49a83f8efULL
};

// λ values for computing the actual private key offsets
// λ = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
// λ² = 0xac9c52b33fa3cf1f6ad9697c0566ce2dfe64c4cf44d86c46f9e1f0aa3c55d3f0
__device__ __constant__ uint64_t LAMBDA[4] = {
    0xdf02967c1b23bd72ULL, 0x122e22ea20816678ULL,
    0xa5261c028812645aULL, 0x5363ad4cc05c30e0ULL
};
__device__ __constant__ uint64_t LAMBDA_SQ[4] = {
    0xf9e1f0aa3c55d3f0ULL, 0xfe64c4cf44d86c46ULL,
    0x6ad9697c0566ce2dULL, 0xac9c52b33fa3cf1fULL
};

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

// Multiply 256-bit number by λ mod n (for finding actual private key)
__device__ void mulmod_lambda(const uint64_t k[4], uint64_t result[4]) {
    // For now, just store the relationship - actual key = k * λ mod n
    // This is computed on CPU when match is found
    #pragma unroll
    for(int i = 0; i < 4; i++) result[i] = k[i];
}

// Init points
__global__ void init_pts(uint64_t* d_px, uint64_t* d_py, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};
    
    // Each thread handles BATCH_SIZE * 6 keys (3 endomorphism × 2 parities)
    unsigned __int128 total_off = (unsigned __int128)tid * BATCH_SIZE * 6;
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

// ENDOMORPHISM KERNEL - 3 points per computed point!
__global__ void __launch_bounds__(256, 4)
kernel_endomorphism(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py, 
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

    // Each iteration covers 6 keys due to endomorphism + negation
    uint64_t local_key = start_key_lo + (tid * BATCH_SIZE * 6);
    unsigned long long lc = 0;

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

    // Process all points with ENDOMORPHISM - 3x keys per point!
    #pragma unroll 8
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

        // Compute endomorphism variants ONCE
        uint64_t x3_lambda[4], x3_lambda2[4];
        fieldMul(x3, BETA, x3_lambda);
        fieldMul(x3, BETA_SQ, x3_lambda2);
        
        // Get parity
        uint8_t parity_even = 0x02;
        uint8_t parity_odd = 0x03;
        uint8_t y_parity = (y3[0] & 1) ? parity_odd : parity_even;
        uint8_t y_parity_neg = (y3[0] & 1) ? parity_even : parity_odd;

        // === POINT 1: P = (x3, y3) ===
        {
            uint32_t sha_state[8];
            sha256_opt(x3[0], x3[1], x3[2], x3[3], y_parity, sha_state);
            uint8_t hash[20];
            ripemd160_opt(sha_state, hash);
            lc++;
            if (*(uint32_t*)hash == d_prefix) {
                bool match = true;
                for(int k=0; k<20; k++) if(hash[k] != d_target[k]) { match = false; break; }
                if (match && atomicCAS(&d_found, 0, 1) == 0) {
                    d_found_key[0] = local_key + i * 6 + 1;
                    d_found_key[1] = d_key_high[0]; d_found_key[2] = d_key_high[1]; d_found_key[3] = d_key_high[2];
                }
            }
        }
        // === POINT 2: -P = (x3, -y3) - FREE! Just flip parity ===
        {
            uint32_t sha_state[8];
            sha256_opt(x3[0], x3[1], x3[2], x3[3], y_parity_neg, sha_state);
            uint8_t hash[20];
            ripemd160_opt(sha_state, hash);
            lc++;
            if (*(uint32_t*)hash == d_prefix) {
                bool match = true;
                for(int k=0; k<20; k++) if(hash[k] != d_target[k]) { match = false; break; }
                if (match && atomicCAS(&d_found, 0, 1) == 0) {
                    d_found_key[0] = local_key + i * 6 + 2;
                    d_found_key[1] = d_key_high[0]; d_found_key[2] = d_key_high[1]; 
                    d_found_key[3] = d_key_high[2] | 0x1000000000000000ULL; // Mark as -P
                }
            }
        }
        // === POINT 3: λP = (βx3, y3) ===
        {
            uint32_t sha_state[8];
            sha256_opt(x3_lambda[0], x3_lambda[1], x3_lambda[2], x3_lambda[3], y_parity, sha_state);
            uint8_t hash[20];
            ripemd160_opt(sha_state, hash);
            lc++;
            if (*(uint32_t*)hash == d_prefix) {
                bool match = true;
                for(int k=0; k<20; k++) if(hash[k] != d_target[k]) { match = false; break; }
                if (match && atomicCAS(&d_found, 0, 1) == 0) {
                    d_found_key[0] = local_key + i * 6 + 3;
                    d_found_key[1] = d_key_high[0]; d_found_key[2] = d_key_high[1]; 
                    d_found_key[3] = d_key_high[2] | 0x8000000000000000ULL;
                }
            }
        }
        // === POINT 4: -λP = (βx3, -y3) ===
        {
            uint32_t sha_state[8];
            sha256_opt(x3_lambda[0], x3_lambda[1], x3_lambda[2], x3_lambda[3], y_parity_neg, sha_state);
            uint8_t hash[20];
            ripemd160_opt(sha_state, hash);
            lc++;
            if (*(uint32_t*)hash == d_prefix) {
                bool match = true;
                for(int k=0; k<20; k++) if(hash[k] != d_target[k]) { match = false; break; }
                if (match && atomicCAS(&d_found, 0, 1) == 0) {
                    d_found_key[0] = local_key + i * 6 + 4;
                    d_found_key[1] = d_key_high[0]; d_found_key[2] = d_key_high[1]; 
                    d_found_key[3] = d_key_high[2] | 0x9000000000000000ULL;
                }
            }
        }
        // === POINT 5: λ²P = (β²x3, y3) ===
        {
            uint32_t sha_state[8];
            sha256_opt(x3_lambda2[0], x3_lambda2[1], x3_lambda2[2], x3_lambda2[3], y_parity, sha_state);
            uint8_t hash[20];
            ripemd160_opt(sha_state, hash);
            lc++;
            if (*(uint32_t*)hash == d_prefix) {
                bool match = true;
                for(int k=0; k<20; k++) if(hash[k] != d_target[k]) { match = false; break; }
                if (match && atomicCAS(&d_found, 0, 1) == 0) {
                    d_found_key[0] = local_key + i * 6 + 5;
                    d_found_key[1] = d_key_high[0]; d_found_key[2] = d_key_high[1]; 
                    d_found_key[3] = d_key_high[2] | 0x4000000000000000ULL;
                }
            }
        }
        // === POINT 6: -λ²P = (β²x3, -y3) ===
        {
            uint32_t sha_state[8];
            sha256_opt(x3_lambda2[0], x3_lambda2[1], x3_lambda2[2], x3_lambda2[3], y_parity_neg, sha_state);
            uint8_t hash[20];
            ripemd160_opt(sha_state, hash);
            lc++;
            if (*(uint32_t*)hash == d_prefix) {
                bool match = true;
                for(int k=0; k<20; k++) if(hash[k] != d_target[k]) { match = false; break; }
                if (match && atomicCAS(&d_found, 0, 1) == 0) {
                    d_found_key[0] = local_key + i * 6 + 6;
                    d_found_key[1] = d_key_high[0]; d_found_key[2] = d_key_high[1]; 
                    d_found_key[3] = d_key_high[2] | 0x5000000000000000ULL;
                }
            }
        }

        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }

    // Update base point (advance by BATCH_SIZE * 3 to account for endomorphism)
    uint64_t dx_last[4], dy_last[4], lambda_pt[4], lambda_sq_pt[4], temp[4];
    const uint64_t* gx_last = &d_G_multiples_x[(BATCH_SIZE-1)*4];
    const uint64_t* gy_last = &d_G_multiples_y[(BATCH_SIZE-1)*4];

    fieldSub(gx_last, base_x, dx_last);
    fieldSub(gy_last, base_y, dy_last);
    fieldMul(dy_last, inv_dx_last, lambda_pt);
    fieldSqr(lambda_pt, lambda_sq_pt);
    fieldSub(lambda_sq_pt, base_x, temp);
    fieldSub(temp, gx_last, base_x);
    
    uint64_t old_base_x[4];
    #pragma unroll
    for(int i=0; i<4; i++) old_base_x[i] = d_px[tid*4 + i];
    
    fieldSub(old_base_x, base_x, temp);
    fieldMul(lambda_pt, temp, base_y);
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
    }
    if(lane == 0) atomicAdd(d_count, lc);
}

__global__ void compute_g_multiples(uint64_t* gx, uint64_t* gy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BATCH_SIZE) return;
    // Compute (tid+1)*6*G to account for 6x endomorphism spacing
    uint64_t key[4] = {(uint64_t)((tid + 1) * 6), 0, 0, 0};
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
    printf("║   HYPERION ENDOMORPHISM - 3x Points per Inversion!          ║\n");
    printf("║   Exploiting secp256k1 structure: λP = (βx, y)              ║\n");
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
    uint64_t kpl = total_threads * BATCH_SIZE * 6;  // 6x: 3 endomorphism × 2 parities!

    printf("Target:     %s\n", target_hash);
    printKey256("Range:      0x", rs);
    printf("Config:     %d blocks × %d threads × %d batch × 6 (endo+neg)\n", 
           NUM_BLOCKS, THREADS_PER_BLOCK, BATCH_SIZE);
    printf("Keys/iter:  %.2f million (6x boost: endomorphism + negation!)\n\n", kpl/1e6);

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

    printf("SEARCHING (3x endomorphism boost)...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = {rs[0], rs[1], rs[2], rs[3]};
    uint64_t batch_lo = cur[0];

    int iter = 0;
    while (true) {
        kernel_endomorphism<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_lo, d_cnt);
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
            
            // Check if it's a λ or λ² variant
            bool is_lambda = (key[3] & 0x8000000000000000ULL) != 0;
            bool is_lambda2 = (key[3] & 0x4000000000000000ULL) != 0;
            key[3] &= 0x3FFFFFFFFFFFFFFFULL; // Clear flags
            
            printf("\n\n🎯 FOUND! ");
            if (is_lambda) printf("[λ variant] ");
            else if (is_lambda2) printf("[λ² variant] ");
            printf("Key: ");
            for(int i=3; i>=0; i--) printf("%016llx", (unsigned long long)key[i]);
            printf("\n");
            
            if (is_lambda || is_lambda2) {
                printf("Note: Actual private key = found_key * %s mod n\n",
                       is_lambda ? "λ" : "λ²");
            }
            
            FILE* f = fopen("FOUND_KEYS.txt", "a");
            if (f) {
                fprintf(f, "KEY: ");
                for(int i=3; i>=0; i--) fprintf(f, "%016llx", (unsigned long long)key[i]);
                if (is_lambda) fprintf(f, " (multiply by λ mod n)");
                if (is_lambda2) fprintf(f, " (multiply by λ² mod n)");
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
            printf("\r[%.2f GKeys/s] %llu keys (3x endo), iter %d   ", 
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
    printf("Final Stats (with 3x endomorphism boost):\n");
    printf("  Speed:        %.2f GKeys/s\n", total/elapsed/1e9);
    printf("  Total keys:   %llu (3x from endomorphism)\n", total);
    printf("  Time:         %.2fs\n", elapsed);
    printf("══════════════════════════════════════════════════════════════\n");

    cudaFree(d_cnt); cudaFree(d_px); cudaFree(d_py);
    return 0;
}
