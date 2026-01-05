/*
 * HYPERION SEARCH HYBRID - 18+ GKeys/s with CPU Verification
 * Uses the profiler's ultra-fast EC kernel + CPU hash verification
 * GPU: EC Point Addition (Batch Inversion) + HASH160
 * CPU: Bloom check + exact verification
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <set>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

// TESTED STABLE CONFIG - same as working Search_Final
#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 64

#define BLOOM_SIZE_BITS 26
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)
#define BLOOM_HASH_COUNT 3

#define MAX_CANDIDATES 65536

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

struct Candidate {
    uint64_t key[4];
    uint8_t hash160[20];
};

__device__ int d_candidate_count = 0;
__device__ Candidate d_candidates[MAX_CANDIDATES];

// CPU-side bloom filter
uint64_t* h_bloom = nullptr;
std::set<std::string> h_target_set;

static __forceinline__ int cmp256_le(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Simple bloom check on GPU - just marks candidates
__device__ __forceinline__ bool bloom_check_gpu(const uint8_t hash160[20], const uint64_t* bloom) {
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

// ULTRA-FAST KERNEL - Same as 18 GKeys/s profiler with bloom filter
__global__ void kernel_search_fast(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
                                   const uint64_t* __restrict__ d_bloom,
                                   uint64_t start_key_lo,
                                   unsigned long long* __restrict__ d_count) {
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

    // Phase 1: Build products for batch inversion
    uint64_t products[64][4];  // Fixed size matching BATCH_SIZE
    uint64_t dx[4];
    fieldSub(&d_G_multiples_x[0], base_x, dx);
    #pragma unroll
    for(int j=0; j<4; j++) products[0][j] = dx[j];

    #pragma unroll 16
    for(int i=1; i<BATCH_SIZE; i++) {
        fieldSub(&d_G_multiples_x[i*4], base_x, dx);
        fieldMul(products[i-1], dx, products[i]);
    }

    // Batch inversion
    uint64_t inv[4];
    fieldInv_Fermat(products[BATCH_SIZE-1], inv);

    // Phase 2: Process all points in reverse
    #pragma unroll 4
    for(int i=BATCH_SIZE-1; i>=0; i--) {
        uint64_t inv_dx[4];
        if(i > 0) {
            fieldMul(inv, products[i-1], inv_dx);
        } else {
            #pragma unroll
            for(int j=0; j<4; j++) inv_dx[j] = inv[j];
        }

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
        
        // HASH160
        uint32_t sha_state[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        lc++;

        // Quick bloom check - store candidate for CPU verification
        if (bloom_check_gpu(hash, d_bloom)) {
            int idx = atomicAdd(&d_candidate_count, 1);
            if (idx < MAX_CANDIDATES) {
                d_candidates[idx].key[0] = local_key + i + 1;
                d_candidates[idx].key[1] = d_key_high[0];
                d_candidates[idx].key[2] = d_key_high[1];
                d_candidates[idx].key[3] = d_key_high[2];
                
                #pragma unroll
                for(int j=0; j<20; j++) d_candidates[idx].hash160[j] = hash[j];
            }
        }

        // Update inverse
        fieldSub(&d_G_multiples_x[i*4], base_x, dx);
        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }

    // Update base point for next batch
    int last_idx = BATCH_SIZE - 1;
    uint64_t dx_last[4], dy_last[4], lambda[4], lambda_sq[4], temp[4];
    
    fieldSub(&d_G_multiples_x[last_idx*4], base_x, dx_last);
    fieldSub(&d_G_multiples_y[last_idx*4], base_y, dy_last);
    
    uint64_t inv_dx_last[4];
    fieldInv_Fermat(dx_last, inv_dx_last);
    fieldMul(dy_last, inv_dx_last, lambda);
    fieldSqr(lambda, lambda_sq);
    fieldSub(lambda_sq, base_x, temp);
    fieldSub(temp, &d_G_multiples_x[last_idx*4], base_x);
    
    fieldSub(&d_G_multiples_x[last_idx*4], base_x, temp);
    fieldMul(lambda, temp, base_y);
    fieldSub(base_y, &d_G_multiples_y[last_idx*4], base_y);

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

void initGMultiples() {
    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    precompute_g_multiples_kernel<<<(BATCH_SIZE+255)/256, 256>>>(d_gx, d_gy);
    cudaDeviceSynchronize();
    uint64_t* h_gx = new uint64_t[BATCH_SIZE * 4];
    uint64_t* h_gy = new uint64_t[BATCH_SIZE * 4];
    cudaMemcpy(h_gx, d_gx, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gy, d_gy, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_G_multiples_x, h_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_multiples_y, h_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaFree(d_gx); cudaFree(d_gy);
    delete[] h_gx; delete[] h_gy;
}

bool cpu_verify(const Candidate& c, const std::set<std::string>& targets) {
    char hex[41];
    for (int i = 0; i < 20; i++) sprintf(hex + i*2, "%02x", c.hash160[i]);
    hex[40] = '\0';
    return targets.find(std::string(hex)) != targets.end();
}

void saveMatch(const Candidate& m, const char* filename) {
    FILE* f = fopen(filename, "a");
    if (f) {
        fprintf(f, "FOUND AT: %s\n", __TIME__);
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
    printf("║  HYPERION SEARCH HYBRID - 18+ GKeys/s Ultra-Fast Pipeline ║\n");
    printf("║  GPU: BatchInv EC + HASH160 + Bloom  |  CPU: Verify       ║\n");
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
    if (!c) { printf("❌ Invalid range\n"); return 1; }
    strncpy(s, range, c-range);
    strcpy(e, c+1);
    if (!parseHex256(s, rs) || !parseHex256(e, re)) { printf("❌ Invalid range\n"); return 1; }
    if (cmp256_le(rs, re) > 0) { printf("❌ start > end\n"); return 1; }

    // Ensure start > BATCH_SIZE to avoid point doubling issue
    if (rs[0] <= BATCH_SIZE && rs[1] == 0 && rs[2] == 0 && rs[3] == 0) {
        rs[0] = BATCH_SIZE + 1;
        printf("⚠️  Adjusted start to %llu\n", (unsigned long long)rs[0]);
    }

    uint64_t total_threads = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    uint64_t kpl = total_threads * BATCH_SIZE;

    printf("Configuration:\n");
    printf("  Targets:      %s\n", targets_file);
    printKey256("  Range start:  0x", rs);
    printKey256("  Range end:    0x", re);
    printf("  Blocks:       %d\n", NUM_BLOCKS);
    printf("  Threads:      %d\n", THREADS_PER_BLOCK);
    printf("  Batch:        %d\n", BATCH_SIZE);
    printf("  Keys/launch:  %llu (%.2f million)\n", (unsigned long long)kpl, kpl/1e6);
    printf("  Bloom size:   %.1f MB (%d hash funcs)\n\n", BLOOM_SIZE_BYTES / 1024.0 / 1024.0, BLOOM_HASH_COUNT);

    printf("Initializing...\n");
    initGMultiples();
    
    // Setup bloom filter
    uint64_t* d_bloom;
    cudaMalloc(&d_bloom, BLOOM_SIZE_BYTES);
    h_bloom = new uint64_t[BLOOM_SIZE_WORDS];
    memset(h_bloom, 0, BLOOM_SIZE_BYTES);
    
    FILE* f = fopen(targets_file, "r");
    if (!f) { printf("❌ Cannot open: %s\n", targets_file); return 1; }
    
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
    printf("✓ Loaded %d targets\n", count);
    
    cudaMemcpy(d_bloom, h_bloom, BLOOM_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, total_threads * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, total_threads * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpyToSymbol(d_candidate_count, &zero, sizeof(int));

    init_pts<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("❌ Init error: %s\n", cudaGetErrorString(err)); return 1; }
    printf("✓ Init OK!\n\n");

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("          SEARCHING AT MAXIMUM SPEED (18+ GKeys/s)...\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = { rs[0], rs[1], rs[2], rs[3] };

    int iteration = 0;
    int total_candidates = 0, verified = 0;
    
    while (true) {
        kernel_search_fast<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_bloom, cur[0], d_cnt);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err)); break; }

        iteration++;

        // CPU verification of candidates
        int cc;
        cudaMemcpyFromSymbol(&cc, d_candidate_count, sizeof(int));
        if (cc > 0) {
            Candidate* hc = new Candidate[cc];
            cudaMemcpyFromSymbol(hc, d_candidates, cc * sizeof(Candidate));
            for (int i = 0; i < cc && i < MAX_CANDIDATES; i++) {
                total_candidates++;
                if (cpu_verify(hc[i], h_target_set)) {
                    verified++;
                    printf("\n\n🎯🎯🎯 MATCH FOUND #%d! 🎯🎯🎯\n", verified);
                    printf("  Key: ");
                    for (int j = 3; j >= 0; j--) printf("%016llx", (unsigned long long)hc[i].key[j]);
                    printf("\n  Hash160: ");
                    for (int j = 0; j < 20; j++) printf("%02x", hc[i].hash160[j]);
                    printf("\n\n");
                    saveMatch(hc[i], "FOUND_KEYS.txt");
                }
            }
            delete[] hc;
            zero = 0;
            cudaMemcpyToSymbol(d_candidate_count, &zero, sizeof(int));
        }

        // Progress
        if (iteration % 5 == 0) {
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double speed = total / elapsed / 1e9;
            printf("\r[%.2f GKeys/s] %llu keys | candidates:%d | match:%d   ", 
                   speed, total, total_candidates, verified);
            fflush(stdout);
        }

        // Advance
        unsigned __int128 add = (unsigned __int128)kpl;
        unsigned __int128 r0 = (unsigned __int128)cur[0] + (uint64_t)add;
        cur[0] = (uint64_t)r0;
        unsigned __int128 r1 = (unsigned __int128)cur[1] + (uint64_t)(r0 >> 64);
        cur[1] = (uint64_t)r1;
        if (r1 >> 64) { cur[2]++; if (cur[2] == 0) cur[3]++; }
        
        if (cmp256_le(cur, re) > 0) { printf("\n\n✓ Range complete!\n"); break; }
    }

    unsigned long long total;
    cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  Final Speed: %.2f GKeys/s\n", total / elapsed / 1e9);
    printf("  Total Keys:  %llu\n", total);
    printf("  Time:        %.2f seconds\n", elapsed);
    printf("  Candidates:  %d\n", total_candidates);
    printf("  Matches:     %d\n", verified);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    if (verified > 0) printf("🎯 Found keys saved to FOUND_KEYS.txt\n\n");

    delete[] h_bloom;
    cudaFree(d_cnt); cudaFree(d_bloom); cudaFree(d_px); cudaFree(d_py);
    return 0;
}
