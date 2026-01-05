/*
 * HYPERION SEARCH DIRECT - Correct but slower version
 * Uses direct scalar multiplication for each key to guarantee correctness
 * Then batch processes hash160 and bloom filter
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <set>
#include <string>

#define NUM_BLOCKS 512
#define THREADS_PER_BLOCK 256
#define KEYS_PER_THREAD 16

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

__device__ __forceinline__ bool bloom_check(const uint8_t hash160[20], const uint64_t* bloom) {
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

// Direct computation kernel - each thread computes KEYS_PER_THREAD keys directly
__global__ void kernel_search_direct(const uint64_t* __restrict__ d_bloom,
                                     uint64_t start0, uint64_t start1, uint64_t start2, uint64_t start3,
                                     unsigned long long* __restrict__ d_count) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31;
    
    unsigned long long lc = 0;
    
    // Compute starting key for this thread
    uint64_t key[4] = {start0, start1, start2, start3};
    unsigned __int128 offset = (unsigned __int128)tid * KEYS_PER_THREAD;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)offset;
    key[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)key[1] + (uint64_t)(r0 >> 64);
    key[1] = (uint64_t)r1;
    if (r1 >> 64) {
        key[2]++;
        if (key[2] == 0) key[3]++;
    }
    
    // Process KEYS_PER_THREAD keys
    #pragma unroll 4
    for (int i = 0; i < KEYS_PER_THREAD; i++) {
        // Direct scalar multiplication - guaranteed correct
        uint64_t px[4], py[4];
        scalarMulBaseAffine(key, px, py);
        
        // Hash160
        uint32_t sha_state[8];
        sha256_opt(px[0], px[1], px[2], px[3], (py[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        lc++;
        
        // Debug: print first few hashes and bloom check result
        if (tid == 0 && i < 5) {
            printf("Key %llu: ", (unsigned long long)key[0]);
            for(int j=0; j<20; j++) printf("%02x", hash[j]);
            bool bc = bloom_check(hash, d_bloom);
            printf(" bloom=%s\n", bc ? "YES" : "no");
        }
        
        // Bloom check
        if (bloom_check(hash, d_bloom)) {
            int idx = atomicAdd(&d_match_count, 1);
            if (idx < MAX_MATCHES) {
                #pragma unroll
                for(int j=0; j<4; j++) d_matches[idx].key[j] = key[j];
                #pragma unroll
                for(int j=0; j<20; j++) d_matches[idx].hash160[j] = hash[j];
            }
        }
        
        // Increment key
        key[0]++;
        if (key[0] == 0) {
            key[1]++;
            if (key[1] == 0) {
                key[2]++;
                if (key[2] == 0) key[3]++;
            }
        }
    }
    
    // Warp reduction for counter
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if(lane == 0) atomicAdd(d_count, lc);
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
    printf("║   HYPERION SEARCH DIRECT - Guaranteed Correct Pipeline    ║\n");
    printf("║  GPU: Direct ScalarMul → HASH160 → Bloom  |  CPU: Verify  ║\n");
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
    uint64_t kpl = total_threads * KEYS_PER_THREAD;

    printf("Configuration:\n");
    printf("  Targets:      %s\n", targets_file);
    printKey256("  Range start:  0x", rs);
    printKey256("  Range end:    0x", re);
    printf("  Blocks:       %d\n", NUM_BLOCKS);
    printf("  Threads:      %d\n", THREADS_PER_BLOCK);
    printf("  Keys/thread:  %d\n", KEYS_PER_THREAD);
    printf("  Keys/batch:   %llu (%.2f million)\n", (unsigned long long)kpl, kpl/1e6);
    printf("  Bloom size:   %.1f MB\n\n", BLOOM_SIZE_BYTES / 1024.0 / 1024.0);

    printf("Initializing...\n");
    
    // Setup bloom filter
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

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpyToSymbol(d_match_count, &zero, sizeof(int));

    printf("✓ Init OK!\n\n");

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                   SEARCHING (DIRECT)...\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = { rs[0], rs[1], rs[2], rs[3] };

    int iteration = 0;
    int total_bloom = 0, verified = 0;
    
    while (true) {
        kernel_search_direct<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_bloom, cur[0], cur[1], cur[2], cur[3], d_cnt);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err)); break; }

        iteration++;

        // Check for matches every iteration for small ranges, or every 10 for large
        if (iteration % 1 == 0) {  // Check every iteration to not miss matches
            int mc;
            cudaMemcpyFromSymbol(&mc, d_match_count, sizeof(int));
            if (mc > 0) {
                Match* hm = new Match[mc];
                cudaMemcpyFromSymbol(hm, d_matches, mc * sizeof(Match));
                for (int i = 0; i < mc && i < MAX_MATCHES; i++) {
                    total_bloom++;
                    if (verifyMatch(hm[i], h_target_set)) {
                        verified++;
                        printf("\n🎯🎯🎯 MATCH FOUND #%d! 🎯🎯🎯\n", verified);
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

        // Print progress
        if (iteration % 50 == 0) {
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
            cur[2]++;
            if (cur[2] == 0) cur[3]++;
        }
        
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

    cudaFree(d_cnt); cudaFree(d_bloom);
    return 0;
}
