/*
 * DEBUG TEST - Verify hash160 calculation matches profiler
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

__global__ void test_hash160_kernel(uint64_t* keys, uint8_t* hashes, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint64_t key[4] = {keys[tid*4], keys[tid*4+1], keys[tid*4+2], keys[tid*4+3]};
    
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    
    uint32_t sha_state[8];
    sha256_opt(ox[0], ox[1], ox[2], ox[3], (oy[0] & 1) ? 0x03 : 0x02, sha_state);
    
    uint8_t h[20];
    ripemd160_opt(sha_state, h);
    
    for(int i=0; i<20; i++) hashes[tid*20 + i] = h[i];
}

int main() {
    printf("=== HASH160 DEBUG TEST ===\n\n");
    
    // Test keys 1, 2, 3, 4, 5
    int count = 5;
    uint64_t h_keys[5][4] = {
        {1, 0, 0, 0},
        {2, 0, 0, 0},
        {3, 0, 0, 0},
        {4, 0, 0, 0},
        {5, 0, 0, 0}
    };
    
    uint64_t* d_keys;
    uint8_t* d_hashes;
    cudaMalloc(&d_keys, count * 4 * sizeof(uint64_t));
    cudaMalloc(&d_hashes, count * 20);
    
    cudaMemcpy(d_keys, h_keys, count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    test_hash160_kernel<<<1, count>>>(d_keys, d_hashes, count);
    cudaDeviceSynchronize();
    
    uint8_t h_hashes[5][20];
    cudaMemcpy(h_hashes, d_hashes, count * 20, cudaMemcpyDeviceToHost);
    
    printf("GPU computed hash160:\n");
    for(int k=0; k<count; k++) {
        printf("  Key %d: ", k+1);
        for(int i=0; i<20; i++) printf("%02x", h_hashes[k][i]);
        printf("\n");
    }
    
    printf("\nExpected (from profiler --samples):\n");
    printf("  Key 1: 751e76e8199196d454941c45d1b3a323f1433bd6\n");
    printf("  Key 2: 06afd46bcdfd22ef94ac122aa11f241244a37ecc\n");
    printf("  Key 3: 7dd65592d0ab2fe0d0257d571abf032cd9db93dc\n");
    printf("  Key 4: c42e7ef92fdb603af844d064faad95db9bcdfd3d\n");
    printf("  Key 5: 4747e8746cddb33b0f7f95a90f89f89fb387cbb6\n");
    
    // Verify
    const char* expected[] = {
        "751e76e8199196d454941c45d1b3a323f1433bd6",
        "06afd46bcdfd22ef94ac122aa11f241244a37ecc",
        "7dd65592d0ab2fe0d0257d571abf032cd9db93dc",
        "c42e7ef92fdb603af844d064faad95db9bcdfd3d",
        "4747e8746cddb33b0f7f95a90f89f89fb387cbb6"
    };
    
    printf("\nVerification:\n");
    int pass = 0;
    for(int k=0; k<count; k++) {
        char computed[41];
        for(int i=0; i<20; i++) sprintf(computed+i*2, "%02x", h_hashes[k][i]);
        computed[40] = 0;
        
        bool match = (strcmp(computed, expected[k]) == 0);
        printf("  Key %d: %s\n", k+1, match ? "✓ PASS" : "✗ FAIL");
        if(match) pass++;
    }
    
    printf("\nResult: %d/%d passed\n", pass, count);
    
    cudaFree(d_keys);
    cudaFree(d_hashes);
    return pass == count ? 0 : 1;
}
