/*
 * DEBUG TEST - Verify kernel processes correct keys and finds targets
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#define BATCH_SIZE 64
#define BLOOM_SIZE_BITS 26
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)
#define BLOOM_HASH_COUNT 5

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

__device__ __constant__ uint64_t d_G_multiples_x[BATCH_SIZE * 4];
__device__ __constant__ uint64_t d_G_multiples_y[BATCH_SIZE * 4];

__device__ __forceinline__ bool bloom_check(const uint8_t hash160[20], const uint64_t* bloom) {
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

// Simplified kernel that just checks first few keys
__global__ void test_kernel(uint64_t base_x[4], uint64_t base_y[4], 
                           const uint64_t* bloom, int* match_count,
                           uint64_t* found_keys, uint8_t* found_hashes) {
    // Process first 10 keys only
    for(int i=0; i<10 && i<BATCH_SIZE; i++) {
        const uint64_t* gx = &d_G_multiples_x[i*4];
        const uint64_t* gy = &d_G_multiples_y[i*4];
        
        // Compute point addition: base + (i+1)*G
        uint64_t dx[4], dy[4];
        fieldSub(gx, base_x, dx);
        fieldSub(gy, base_y, dy);
        
        uint64_t inv_dx[4];
        fieldInv_Fermat(dx, inv_dx);
        
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
        
        // Hash
        uint32_t sha_state[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        // Debug: store first 5 results
        if(i < 5) {
            for(int j=0; j<20; j++) found_hashes[i*20 + j] = hash[j];
            found_keys[i] = i + 2;  // Key = base_key(1) + (i+1) = i+2
        }
        
        // Check bloom
        if(bloom_check(hash, bloom)) {
            int idx = atomicAdd(match_count, 1);
            printf("GPU: Key %d matched bloom! hash=", i+2);
            for(int j=0; j<20; j++) printf("%02x", hash[j]);
            printf("\n");
        }
    }
}

__global__ void precompute_g_multiples(uint64_t* gx, uint64_t* gy) {
    int tid = threadIdx.x;
    if (tid >= BATCH_SIZE) return;
    uint64_t key[4] = {(uint64_t)(tid + 1), 0, 0, 0};
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    for(int i=0; i<4; i++) {
        gx[tid*4 + i] = ox[i];
        gy[tid*4 + i] = oy[i];
    }
}

int main() {
    printf("=== KERNEL DEBUG TEST ===\n\n");
    
    // Precompute G multiples
    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    precompute_g_multiples<<<1, BATCH_SIZE>>>(d_gx, d_gy);
    cudaDeviceSynchronize();
    
    uint64_t* h_gx = new uint64_t[BATCH_SIZE * 4];
    uint64_t* h_gy = new uint64_t[BATCH_SIZE * 4];
    cudaMemcpy(h_gx, d_gx, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gy, d_gy, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_G_multiples_x, h_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_multiples_y, h_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaFree(d_gx); cudaFree(d_gy);
    
    // Compute base point = 1*G
    uint64_t h_base_x[4], h_base_y[4];
    uint64_t key1[4] = {1, 0, 0, 0};
    uint64_t *d_key, *d_ox, *d_oy;
    cudaMalloc(&d_key, 32); cudaMalloc(&d_ox, 32); cudaMalloc(&d_oy, 32);
    cudaMemcpy(d_key, key1, 32, cudaMemcpyHostToDevice);
    
    // Use existing scalarMulBaseAffine
    extern __global__ void scalarMulKernelBase(const uint64_t* k, uint64_t* ox, uint64_t* oy, int count);
    scalarMulKernelBase<<<1, 1>>>(d_key, d_ox, d_oy, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_base_x, d_ox, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_base_y, d_oy, 32, cudaMemcpyDeviceToHost);
    cudaFree(d_key); cudaFree(d_ox); cudaFree(d_oy);
    
    printf("Base point (1*G):\n  X: ");
    for(int i=3; i>=0; i--) printf("%016llx", (unsigned long long)h_base_x[i]);
    printf("\n  Y: ");
    for(int i=3; i>=0; i--) printf("%016llx", (unsigned long long)h_base_y[i]);
    printf("\n\n");
    
    // Build bloom filter with targets for keys 2, 3, 4
    const char* targets[] = {
        "06afd46bcdfd22ef94ac122aa11f241244a37ecc",  // key 2
        "7dd65592d0ab2fe0d0257d571abf032cd9db93dc",  // key 3
        "c42e7ef92fdb603af844d064faad95db9bcdfd3d",  // key 4
    };
    
    uint64_t* h_bloom = new uint64_t[BLOOM_SIZE_WORDS];
    memset(h_bloom, 0, BLOOM_SIZE_BYTES);
    
    printf("Targets in bloom filter:\n");
    for(int t=0; t<3; t++) {
        uint8_t hash160[20];
        parseHash160(targets[t], hash160);
        printf("  Key %d: %s\n", t+2, targets[t]);
        
        uint64_t seed = *(uint64_t*)hash160;
        uint64_t h1 = seed, h2 = seed * 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < BLOOM_HASH_COUNT; i++) {
            uint64_t hash = h1 + i * h2;
            uint64_t bit_idx = hash & ((1ULL << BLOOM_SIZE_BITS) - 1);
            h_bloom[bit_idx >> 6] |= (1ULL << (bit_idx & 63));
        }
    }
    
    uint64_t* d_bloom;
    cudaMalloc(&d_bloom, BLOOM_SIZE_BYTES);
    cudaMemcpy(d_bloom, h_bloom, BLOOM_SIZE_BYTES, cudaMemcpyHostToDevice);
    
    // Allocate output
    uint64_t* d_base_x; uint64_t* d_base_y;
    cudaMalloc(&d_base_x, 32); cudaMalloc(&d_base_y, 32);
    cudaMemcpy(d_base_x, h_base_x, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_y, h_base_y, 32, cudaMemcpyHostToDevice);
    
    int* d_match_count;
    uint64_t* d_found_keys;
    uint8_t* d_found_hashes;
    cudaMalloc(&d_match_count, sizeof(int));
    cudaMalloc(&d_found_keys, 5 * sizeof(uint64_t));
    cudaMalloc(&d_found_hashes, 5 * 20);
    int zero = 0;
    cudaMemcpy(d_match_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("\nRunning kernel (base=1, processing keys 2-11)...\n\n");
    
    test_kernel<<<1, 1>>>(d_base_x, d_base_y, d_bloom, d_match_count, d_found_keys, d_found_hashes);
    cudaDeviceSynchronize();
    
    // Get results
    int match_count;
    uint64_t found_keys[5];
    uint8_t found_hashes[5][20];
    cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_keys, d_found_keys, 5 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_hashes, d_found_hashes, 5 * 20, cudaMemcpyDeviceToHost);
    
    printf("Computed hashes for first 5 keys:\n");
    for(int i=0; i<5; i++) {
        printf("  Key %llu: ", (unsigned long long)found_keys[i]);
        for(int j=0; j<20; j++) printf("%02x", found_hashes[i][j]);
        printf("\n");
    }
    
    printf("\nBloom matches found: %d\n", match_count);
    printf("Expected: 3 (keys 2, 3, 4)\n");
    
    delete[] h_bloom;
    delete[] h_gx;
    delete[] h_gy;
    
    return match_count == 3 ? 0 : 1;
}
