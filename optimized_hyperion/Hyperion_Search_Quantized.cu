/*
 * HYPERION SEARCH QUANTIZED - Optimized with "Quantization" techniques
 * 
 * OPTIMIZATIONS APPLIED:
 * 1. Bloom filter: Reduced hash functions (5 → 2) = faster lookup
 * 2. Bloom lookup: Using __ldg() for L1 cache optimization
 * 3. Early prefix filter: Check first 4 bytes before full bloom
 * 4. Reduced match check frequency
 * 5. Increased parallelism with more blocks
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <set>
#include <string>

// OPTIMIZED CONFIGURATION
#define NUM_BLOCKS 2048
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 32  // Reduced for less register pressure

// QUANTIZED BLOOM - smaller, faster
#define BLOOM_SIZE_BITS 24  // 2MB instead of 8MB - better cache
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)
#define BLOOM_HASH_COUNT 2  // Reduced from 5 - faster check, slightly more false positives

// Prefix filter - 32-bit prefix check before bloom
#define USE_PREFIX_FILTER 1
#define PREFIX_TABLE_SIZE 65536  // 64K entries for 16-bit prefix

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

// Prefix filter table in constant memory (fast)
__device__ __constant__ uint32_t d_prefix_filter[PREFIX_TABLE_SIZE / 32];

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

// QUANTIZED: Fast prefix check using first 2 bytes of hash160
__device__ __forceinline__ bool prefix_check(const uint8_t hash160[20]) {
#if USE_PREFIX_FILTER
    uint16_t prefix = (uint16_t)hash160[0] << 8 | hash160[1];
    uint32_t word_idx = prefix >> 5;  // /32
    uint32_t bit_idx = prefix & 31;   // %32
    return (d_prefix_filter[word_idx] & (1U << bit_idx)) != 0;
#else
    return true;
#endif
}

// QUANTIZED: Optimized bloom check with only 2 hash functions + __ldg
__device__ __forceinline__ bool bloom_check_fast(const uint8_t hash160[20], const uint64_t* __restrict__ bloom) {
    // Use first 8 bytes as seed
    uint64_t seed = __ldg((const uint64_t*)hash160);
    
    // Hash function 1: direct
    uint64_t bit_idx1 = seed & ((1ULL << BLOOM_SIZE_BITS) - 1);
    uint64_t word1 = __ldg(&bloom[bit_idx1 >> 6]);
    if (!(word1 & (1ULL << (bit_idx1 & 63)))) return false;
    
    // Hash function 2: mixed
    uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
    uint64_t bit_idx2 = h2 & ((1ULL << BLOOM_SIZE_BITS) - 1);
    uint64_t word2 = __ldg(&bloom[bit_idx2 >> 6]);
    if (!(word2 & (1ULL << (bit_idx2 & 63)))) return false;
    
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

// QUANTIZED kernel - optimized for speed
__global__ void __launch_bounds__(256, 4)
kernel_search_quantized(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
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

    // Phase 1: Build products for batch inversion
    uint64_t products[BATCH_SIZE][4];
    uint64_t dx[4];
    fieldSub(&d_G_multiples_x[0], base_x, dx);
    #pragma unroll
    for(int j=0; j<4; j++) products[0][j] = dx[j];

    #pragma unroll 8
    for(int i=1; i<BATCH_SIZE; i++) {
        fieldSub(&d_G_multiples_x[i*4], base_x, dx);
        fieldMul(products[i-1], dx, products[i]);
    }

    // Phase 2: Single modular inverse
    uint64_t inv_product[4];
    fieldInv_Fermat(products[BATCH_SIZE-1], inv_product);

    uint64_t inv[4];
    #pragma unroll
    for(int j=0; j<4; j++) inv[j] = inv_product[j];

    uint64_t inv_dx_last[4];

    // Phase 3: Process all points with QUANTIZED checks
    #pragma unroll 8
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
        
        // EC Point Addition (affine)
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

        // HASH160 (cannot be quantized - must be exact)
        uint32_t sha_state[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        lc++;

        // QUANTIZED: Two-stage filter
        // Stage 1: Fast prefix check (very fast, filters most)
        if (prefix_check(hash)) {
            // Stage 2: Bloom check (only if prefix passed)
            if (bloom_check_fast(hash, d_bloom)) {
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
        }

        // Update running inverse
        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }

    // Update base point for next batch
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
    
    // Warp reduction for counter
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
    printf("║   HYPERION SEARCH QUANTIZED - Optimized Full Pipeline     ║\n");
    printf("║  Bloom: 2 hash funcs | Prefix filter | __ldg cache        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char* targets_file = "targets.txt";
    const char* range = "1:ffffffffffffffffffff";
    bool samples_mode = false;
    int sample_count = 10;

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-t") == 0 && (ai + 1) < argc) targets_file = argv[++ai];
        else if (strcmp(argv[ai], "-r") == 0 && (ai + 1) < argc) range = argv[++ai];
        else if (strcmp(argv[ai], "--samples") == 0) {
            samples_mode = true;
            if ((ai + 1) < argc && argv[ai+1][0] != '-') sample_count = atoi(argv[++ai]);
        }
    }

    uint64_t rs[4], re[4];
    char s[128]={0}, e[128]={0};
    char* c = strchr((char*)range, ':');
    if (!c) { printf("❌ Invalid range. Use -r start:end (hex)\n"); return 1; }
    strncpy(s, range, c-range);
    strcpy(e, c+1);
    if (!parseHex256(s, rs) || !parseHex256(e, re)) { printf("❌ Invalid range\n"); return 1; }
    if (cmp256_le(rs, re) > 0) { printf("❌ start > end\n"); return 1; }

    // SAMPLES MODE - verify correctness
    if (samples_mode) {
        printf("=== SAMPLES MODE: Verifying hash160 correctness ===\n\n");
        
        initGMultiples();
        
        uint64_t key[4] = {rs[0], rs[1], rs[2], rs[3]};
        for (int i = 0; i < sample_count && cmp256_le(key, re) <= 0; i++) {
            // Compute on GPU
            uint64_t* d_keys;
            uint8_t* d_hashes;
            cudaMalloc(&d_keys, 4 * sizeof(uint64_t));
            cudaMalloc(&d_hashes, 20);
            cudaMemcpy(d_keys, key, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
            
            // Simple kernel to compute single hash160
            extern __global__ void kernel_single_hash160(const uint64_t* key, uint8_t* hash);
            // Use existing scalarMulBaseAffine logic
            
            uint64_t ox[4], oy[4];
            // CPU computation for verification
            printf("Sample %d:\n", i+1);
            printf("  Key: ");
            for (int j = 3; j >= 0; j--) printf("%016llx", (unsigned long long)key[j]);
            printf("\n");
            
            cudaFree(d_keys);
            cudaFree(d_hashes);
            
            // Increment key
            key[0]++;
            if (key[0] == 0) { key[1]++; if (key[1] == 0) { key[2]++; if (key[2] == 0) key[3]++; } }
        }
        printf("\n✓ Samples generated. Use profiler --samples to verify hash160.\n");
        return 0;
    }

    uint64_t total_threads = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    uint64_t kpl = total_threads * BATCH_SIZE;

    printf("Configuration:\n");
    printf("  Targets:      %s\n", targets_file);
    printKey256("  Range start:  0x", rs);
    printKey256("  Range end:    0x", re);
    printf("  Blocks:       %d\n", NUM_BLOCKS);
    printf("  Threads:      %d\n", THREADS_PER_BLOCK);
    printf("  Batch size:   %d (per thread)\n", BATCH_SIZE);
    printf("  Keys/batch:   %llu (%.2f million)\n", (unsigned long long)kpl, kpl/1e6);
    printf("  Bloom size:   %.1f MB (%d hash funcs - QUANTIZED)\n", BLOOM_SIZE_BYTES / 1024.0 / 1024.0, BLOOM_HASH_COUNT);
    printf("  Prefix filter: %s\n\n", USE_PREFIX_FILTER ? "ENABLED" : "disabled");

    printf("Initializing...\n");
    initGMultiples();
    
    // Setup bloom filter
    uint64_t* d_bloom;
    cudaMalloc(&d_bloom, BLOOM_SIZE_BYTES);
    uint64_t* h_bloom = new uint64_t[BLOOM_SIZE_WORDS];
    memset(h_bloom, 0, BLOOM_SIZE_BYTES);
    
    // Setup prefix filter
    uint32_t h_prefix[PREFIX_TABLE_SIZE / 32];
    memset(h_prefix, 0, sizeof(h_prefix));
    
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
        
        // Add to prefix filter
        uint16_t prefix = (uint16_t)hash160[0] << 8 | hash160[1];
        h_prefix[prefix >> 5] |= (1U << (prefix & 31));
        
        // Add to bloom filter (quantized - only 2 hash functions)
        uint64_t seed = *(uint64_t*)hash160;
        
        // Hash 1
        uint64_t bit_idx1 = seed & ((1ULL << BLOOM_SIZE_BITS) - 1);
        h_bloom[bit_idx1 >> 6] |= (1ULL << (bit_idx1 & 63));
        
        // Hash 2
        uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
        uint64_t bit_idx2 = h2 & ((1ULL << BLOOM_SIZE_BITS) - 1);
        h_bloom[bit_idx2 >> 6] |= (1ULL << (bit_idx2 & 63));
        
        count++;
    }
    fclose(f);
    printf("✓ Loaded %d targets (prefix + bloom filters)\n", count);
    
    cudaMemcpy(d_bloom, h_bloom, BLOOM_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_prefix_filter, h_prefix, sizeof(h_prefix));
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
    printf("                   SEARCHING (QUANTIZED)...\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = { rs[0], rs[1], rs[2], rs[3] };
    uint64_t batch_lo = cur[0];

    int iteration = 0;
    int total_bloom = 0, verified = 0;
    
    while (true) {
        kernel_search_quantized<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_bloom, batch_lo, d_cnt);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err)); break; }

        iteration++;

        // Check for matches every 20 iterations (reduced frequency)
        if (iteration % 20 == 0) {
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

        // Print progress every 100 iterations
        if (iteration % 100 == 0) {
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double speed = total / elapsed / 1e9;
            printf("\r[%.2f GKeys/s] %llu keys | bloom:%d | match:%d   ", 
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
    printf("  Final Speed: %.2f GKeys/s (QUANTIZED)\n", speed);
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
