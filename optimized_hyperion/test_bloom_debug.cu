/*
 * DEBUG TEST - Verify bloom filter logic
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#define BLOOM_SIZE_BITS 26
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)
#define BLOOM_HASH_COUNT 5

#include "CUDAUtils.h"

__device__ __forceinline__ bool bloom_check_device(const uint8_t hash160[20], const uint64_t* bloom) {
    uint64_t seed = *(uint64_t*)hash160;
    uint64_t h1 = seed;
    uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
    
    for (int i = 0; i < BLOOM_HASH_COUNT; i++) {
        uint64_t hash = h1 + i * h2;
        uint64_t bit_idx = hash & ((1ULL << BLOOM_SIZE_BITS) - 1);
        if (!(bloom[bit_idx >> 6] & (1ULL << (bit_idx & 63)))) return false;
    }
    return true;
}

__global__ void test_bloom_kernel(const uint64_t* bloom, const uint8_t* hashes, int count, int* results) {
    int tid = threadIdx.x;
    if (tid >= count) return;
    results[tid] = bloom_check_device(&hashes[tid*20], bloom) ? 1 : 0;
}

int main() {
    printf("=== BLOOM FILTER DEBUG TEST ===\n\n");
    
    // Target hashes
    const char* targets[] = {
        "751e76e8199196d454941c45d1b3a323f1433bd6",  // key 1
        "06afd46bcdfd22ef94ac122aa11f241244a37ecc",  // key 2
        "7dd65592d0ab2fe0d0257d571abf032cd9db93dc",  // key 3
    };
    int num_targets = 3;
    
    // Build bloom filter on host (same as Hyperion_Search_Final)
    uint64_t* h_bloom = new uint64_t[BLOOM_SIZE_WORDS];
    memset(h_bloom, 0, BLOOM_SIZE_BYTES);
    
    printf("Building bloom filter with %d targets:\n", num_targets);
    for(int t=0; t<num_targets; t++) {
        uint8_t hash160[20];
        parseHash160(targets[t], hash160);
        
        uint64_t seed = *(uint64_t*)hash160;
        uint64_t h1 = seed;
        uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
        
        printf("  Target %d: %s\n", t+1, targets[t]);
        printf("    Seed: %016llx\n", (unsigned long long)seed);
        printf("    Bits set: ");
        
        for (int i = 0; i < BLOOM_HASH_COUNT; i++) {
            uint64_t hash = h1 + i * h2;
            uint64_t bit_idx = hash & ((1ULL << BLOOM_SIZE_BITS) - 1);
            h_bloom[bit_idx >> 6] |= (1ULL << (bit_idx & 63));
            printf("%llu ", (unsigned long long)bit_idx);
        }
        printf("\n");
    }
    
    // Copy to device
    uint64_t* d_bloom;
    cudaMalloc(&d_bloom, BLOOM_SIZE_BYTES);
    cudaMemcpy(d_bloom, h_bloom, BLOOM_SIZE_BYTES, cudaMemcpyHostToDevice);
    
    // Test hashes on device
    uint8_t h_hashes[3][20];
    for(int t=0; t<num_targets; t++) {
        parseHash160(targets[t], h_hashes[t]);
    }
    
    uint8_t* d_hashes;
    int* d_results;
    cudaMalloc(&d_hashes, num_targets * 20);
    cudaMalloc(&d_results, num_targets * sizeof(int));
    cudaMemcpy(d_hashes, h_hashes, num_targets * 20, cudaMemcpyHostToDevice);
    
    test_bloom_kernel<<<1, num_targets>>>(d_bloom, d_hashes, num_targets, d_results);
    cudaDeviceSynchronize();
    
    int h_results[3];
    cudaMemcpy(h_results, d_results, num_targets * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nBloom filter check results:\n");
    int pass = 0;
    for(int t=0; t<num_targets; t++) {
        printf("  Target %d: %s\n", t+1, h_results[t] ? "✓ FOUND in bloom" : "✗ NOT found");
        if(h_results[t]) pass++;
    }
    
    printf("\nResult: %d/%d bloom checks passed\n", pass, num_targets);
    
    delete[] h_bloom;
    cudaFree(d_bloom);
    cudaFree(d_hashes);
    cudaFree(d_results);
    
    return pass == num_targets ? 0 : 1;
}
