/*
 * HYPERION SEARCH FINAL - 16+ GKeys/s Full Pipeline
 * Based on the proven profiler kernel with bloom filter for multiple targets
 * 
 * PIPELINE:
 * [GPU] privkey++ → Affine EC add (batch inv 256) → HASH160 → Bloom → buffer
 * [CPU] Base58/Bech32 only for matches
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <set>
#include <string>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

#define BLOOM_SIZE_BITS 26
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)
#define BLOOM_HASH_COUNT 5

#define MAX_MATCHES 4096

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

struct Match {
    uint64_t key[4];
    uint8_t hash160[20];
};

__device__ int d_match_count = 0;
__device__ Match d_matches[MAX_MATCHES];

std::set<std::string> h_target_set;

static __forceinline__ int cmp256_le(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ __forceinline__ bool bloom_check(const uint8_t hash160[20], const uint64_t* __restrict__ bloom) {
    uint64_t seed = *(uint64_t*)hash160;
    uint64_t h1 = seed;
    uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
    
    #pragma unroll
    for (int i = 0; i < BLOOM_HASH_COUNT; i++) {
        uint64_t hash = h1 + i * h2;
        uint64_t bit_idx = hash & ((1ULL << BLOOM_SIZE_BITS) - 1);
        if (!(bloom[bit_idx >> 6] & (1ULL << (bit_idx & 63)))) return false;
    }
    return true;
}

__global__ void precompute_g_multiples_kernel(uint64_t* gx, uint64_t* gy) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= BATCH_SIZE) return;
    uint64_t key[4] = {(uint64_t)(tid + 1), 0, 0, 0};
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    #pragma unroll
    for(int i=0; i<4; i++) {
        gx[tid*4 + i] = ox[i];
        gy[tid*4 + i] = oy[i];
    }
}

__global__ void init_pts(uint64_t* d_px, uint64_t* d_py, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};
    
    unsigned __int128 total_off = (unsigned __int128)tid * BATCH_SIZE;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)total_off;
    key[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)key[1] + (uint64_t)(r0 >> 64);
    key[1] = (uint64_t)r1;
    if (r1 >> 64) {
        uint64_t r2 = key[2] + 1;
        key[2] = r2;
        if (r2 == 0) key[3]++;
    }
    
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    
    #pragma unroll
    for(int i=0; i<4; i++) {
        d_px[tid*4 + i] = ox[i];
        d_py[tid*4 + i] = oy[i];
    }
}

// Main search kernel - exact copy of profiler kernel with bloom filter
__global__ void __launch_bounds__(256, 4)
kernel_search_final(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
                    const uint64_t* __restrict__ d_bloom,
                    uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31;

    uint64_t base_x[4], base_y[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        base_x[i] = d_px[tid*4 + i];
        base_y[i] = d_py[tid*4 + i];
    }

    uint64_t local_key = start_key_lo + (tid * BATCH_SIZE);
    unsigned long long lc = 0;

    // Phase 1: Build products for batch inversion (same as profiler)
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

    // Phase 2: Single modular inverse (same as profiler)
    uint64_t inv_product[4];
    fieldInv_Fermat(products[BATCH_SIZE-1], inv_product);

    uint64_t inv[4];
    #pragma unroll
    for(int j=0; j<4; j++) inv[j] = inv_product[j];

    uint64_t inv_dx_last[4];

    // Phase 3: Process all points (same as profiler but with bloom check)
    #pragma unroll 16
    for(int i=BATCH_SIZE-1; i>=0; i--) {
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
        
        // EC Point Addition (affine) - same as profiler
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

        // HASH160 - same as profiler
        uint32_t sha_state[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        lc++;

        // BLOOM CHECK instead of single target check
        if (bloom_check(hash, d_bloom)) {
            int idx = atomicAdd(&d_match_count, 1);
            if (idx < MAX_MATCHES) {
                d_matches[idx].key[0] = local_key + i + 1;
                d_matches[idx].key[1] = d_key_high[0];
                d_matches[idx].key[2] = d_key_high[1];
                d_matches[idx].key[3] = d_key_high[2];
                
                #pragma unroll
                for(int j=0; j<20; j++) d_matches[idx].hash160[j] = hash[j];
            }
        }

        // Update running inverse - same as profiler
        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }

    // Update base point for next batch - same as profiler
    int last_idx = BATCH_SIZE - 1;
    uint64_t dx_last[4], dy_last[4], lambda[4], lambda_sq[4], temp[4];
    
    const uint64_t* gx_last = &d_G_multiples_x[last_idx*4];
    const uint64_t* gy_last = &d_G_multiples_y[last_idx*4];

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
    
    // Warp reduction for counter - same as profiler
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if(lane == 0) atomicAdd(d_count, lc);
}

void initGMultiples() {
    uint64_t* h_gx = new uint64_t[BATCH_SIZE * 4];
    uint64_t* h_gy = new uint64_t[BATCH_SIZE * 4];
    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    precompute_g_multiples_kernel<<<1, BATCH_SIZE>>>(d_gx, d_gy);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gx, d_gx, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gy, d_gy, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaMemcpyToSymbol(d_G_multiples_x, h_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_multiples_y, h_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    delete[] h_gx;
    delete[] h_gy;
}

static void printHash160(const char* label, const uint8_t h[20]) {
    printf("%s", label);
    for (int i = 0; i < 20; i++) printf("%02x", (unsigned)h[i]);
    printf("\n");
}

static void printKey256Hex(const char* label, const uint64_t k[4]) {
    printf("%s", label);
    for (int i = 3; i >= 0; i--) printf("%016llx", (unsigned long long)k[i]);
    printf("\n");
}

bool verifyMatch(const Match& m, const std::set<std::string>& targets) {
    char hex[41];
    for (int i = 0; i < 20; i++) sprintf(hex + i*2, "%02x", m.hash160[i]);
    hex[40] = '\0';
    return targets.find(std::string(hex)) != targets.end();
}

void saveMatch(const Match& m, const char* filename) {
    FILE* f = fopen(filename, "a");
    if (f) {
        fprintf(f, "KEY: ");
        for (int i = 3; i >= 0; i--) fprintf(f, "%016llx", (unsigned long long)m.key[i]);
        fprintf(f, "\nHASH160: ");
        for (int i = 0; i < 20; i++) fprintf(f, "%02x", m.hash160[i]);
        fprintf(f, "\n\n");
        fclose(f);
    }
}

int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     HYPERION SEARCH FINAL - 16+ GKeys/s Full Pipeline     ║\n");
    printf("║  GPU: Affine BatchInv → HASH160 → Bloom  |  CPU: Verify   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char* targets_file = "targets.txt";
    const char* range = "1:ffffffffffffffffffff";

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-t") == 0 && (ai + 1) < argc) targets_file = argv[++ai];
        else if (strcmp(argv[ai], "-r") == 0 && (ai + 1) < argc) range = argv[++ai];
    }

    uint64_t rs[4], re[4];
    char s[128]={0}, e[128]={0};
    char* c = strchr((char*)range, ':');
    if (!c) { printf("❌ Invalid range. Use -r start:end (hex)\n"); return 1; }
    strncpy(s, range, c-range);
    strcpy(e, c+1);
    if (!parseHex256(s, rs) || !parseHex256(e, re)) { printf("❌ Invalid range\n"); return 1; }
    if (cmp256_le(rs, re) > 0) { printf("❌ start > end\n"); return 1; }

    uint64_t total_threads = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    uint64_t kpl = total_threads * BATCH_SIZE;

    printf("Configuration:\n");
    printf("  Targets:      %s\n", targets_file);
    printKey256("  Range start:  0x", rs);
    printKey256("  Range end:    0x", re);
    printf("  Blocks:       %d\n", NUM_BLOCKS);
    printf("  Threads:      %d\n", THREADS_PER_BLOCK);
    printf("  Keys/batch:   %llu (%.2f million)\n", (unsigned long long)kpl, kpl/1e6);
    printf("  Bloom size:   %.1f MB\n\n", BLOOM_SIZE_BYTES / 1024.0 / 1024.0);

    printf("Initializing...\n");
    initGMultiples();
    
    uint64_t* d_bloom;
    cudaMalloc(&d_bloom, BLOOM_SIZE_BYTES);
    uint64_t* h_bloom = new uint64_t[BLOOM_SIZE_WORDS];
    memset(h_bloom, 0, BLOOM_SIZE_BYTES);
    
    FILE* f = fopen(targets_file, "r");
    if (!f) {
        printf("❌ Cannot open targets file: %s\n", targets_file);
        return 1;
    }
    
    char line[128];
    int count = 0;
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (strlen(line) != 40) continue;
        uint8_t hash160[20];
        if (!parseHash160(line, hash160)) continue;
        h_target_set.insert(std::string(line));
        uint64_t seed = *(uint64_t*)hash160;
        uint64_t h1 = seed, h2 = seed * 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < BLOOM_HASH_COUNT; i++) {
            uint64_t hash = h1 + i * h2;
            uint64_t bit_idx = hash & ((1ULL << BLOOM_SIZE_BITS) - 1);
            h_bloom[bit_idx >> 6] |= (1ULL << (bit_idx & 63));
        }
        count++;
    }
    fclose(f);
    printf("✓ Loaded %d targets into bloom filter\n", count);
    
    cudaMemcpy(d_bloom, h_bloom, BLOOM_SIZE_BYTES, cudaMemcpyHostToDevice);
    delete[] h_bloom;

    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, total_threads * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, total_threads * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpyToSymbol(d_match_count, &zero, sizeof(int));

    init_pts<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("❌ Init error: %s\n", cudaGetErrorString(err)); return 1; }
    printf("✓ Init OK!\n\n");

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                      SEARCHING...\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = { rs[0], rs[1], rs[2], rs[3] };
    uint64_t batch_lo = cur[0];

    int iteration = 0;
    int total_bloom = 0, verified = 0;
    
    while (true) {
        kernel_search_final<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_bloom, batch_lo, d_cnt);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err)); break; }

        iteration++;

        // Check for matches every 10 iterations
        if (iteration % 10 == 0) {
            int mc;
            cudaMemcpyFromSymbol(&mc, d_match_count, sizeof(int));
            if (mc > 0) {
                Match* hm = new Match[mc];
                cudaMemcpyFromSymbol(hm, d_matches, mc * sizeof(Match));
                for (int i = 0; i < mc && i < MAX_MATCHES; i++) {
                    total_bloom++;
                    if (verifyMatch(hm[i], h_target_set)) {
                        verified++;
                        printf("\n");
                        printf("🎯🎯🎯 MATCH FOUND #%d! 🎯🎯🎯\n", verified);
                        printKey256Hex("  Private Key: ", hm[i].key);
                        printHash160("  Hash160:     ", hm[i].hash160);
                        printf("\n");
                        saveMatch(hm[i], "FOUND_KEYS.txt");
                    }
                }
                delete[] hm;
                zero = 0;
                cudaMemcpyToSymbol(d_match_count, &zero, sizeof(int));
            }
        }

        // Print progress every 50 iterations
        if (iteration % 50 == 0) {
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double speed = total / elapsed / 1e9;
            printf("\r[%.2f GKeys/s] %llu keys checked | bloom:%d | confirmed:%d   ", 
                   speed, total, total_bloom, verified);
            fflush(stdout);
        }

        // Advance range
        unsigned __int128 add = (unsigned __int128)kpl;
        unsigned __int128 r0 = (unsigned __int128)cur[0] + (uint64_t)add;
        cur[0] = (uint64_t)r0;
        unsigned __int128 r1 = (unsigned __int128)cur[1] + (uint64_t)(r0 >> 64);
        cur[1] = (uint64_t)r1;
        if (r1 >> 64) {
            unsigned __int128 r2 = (unsigned __int128)cur[2] + 1;
            cur[2] = (uint64_t)r2;
            if (r2 >> 64) cur[3]++;
        }
        
        batch_lo = cur[0];
        cudaMemcpyToSymbol(d_key_high, &cur[1], 24);
        
        if (cmp256_le(cur, re) > 0) { 
            printf("\n\n✓ Range complete!\n"); 
            break; 
        }
    }

    unsigned long long total;
    cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double speed = total / elapsed / 1e9;

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  Final Speed: %.2f GKeys/s\n", speed);
    printf("  Total Keys:  %llu\n", total);
    printf("  Time:        %.2f seconds\n", elapsed);
    printf("  Bloom hits:  %d\n", total_bloom);
    printf("  Confirmed:   %d\n", verified);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    if (verified > 0) {
        printf("🎯 Found keys saved to FOUND_KEYS.txt\n\n");
    }

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_cnt); cudaFree(d_bloom);
    return 0;
}
