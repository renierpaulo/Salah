/*
 * HYPERION JACOBIAN - Based on profiler's 16+ GKeys/s architecture
 * Key insight: 1 base point per BLOCK in shared memory (Jacobian)
 * All threads add their own G multiple - NO batch inversion needed!
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define STEPS_PER_KERNEL 256

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

__device__ __constant__ uint64_t d_key_high[3];
__device__ __constant__ uint64_t d_G_multiples_x[THREADS_PER_BLOCK * 4];
__device__ __constant__ uint64_t d_G_multiples_y[THREADS_PER_BLOCK * 4];
__device__ __constant__ uint64_t d_G_step_x[4]; // THREADS_PER_BLOCK * G for advancing
__device__ __constant__ uint64_t d_G_step_y[4];
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

// Jacobian point addition: R = P + Q where P is Jacobian, Q is affine
__device__ __forceinline__ void pointAddMixed(
    const uint64_t Px[4], const uint64_t Py[4], const uint64_t Pz[4],
    const uint64_t Qx[4], const uint64_t Qy[4],
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4]) {
    
    uint64_t Z2[4], U2[4], S2[4], H[4], I[4], J[4], r[4], V[4], T[4];
    
    // Z2 = Pz^2
    fieldSqr(Pz, Z2);
    // U2 = Qx * Z2
    fieldMul(Qx, Z2, U2);
    // S2 = Qy * Pz * Z2
    fieldMul(Pz, Z2, T);
    fieldMul(Qy, T, S2);
    // H = U2 - Px
    fieldSub(U2, Px, H);
    // I = (2*H)^2
    fieldAdd(H, H, T);
    fieldSqr(T, I);
    // J = H * I
    fieldMul(H, I, J);
    // r = 2*(S2 - Py)
    fieldSub(S2, Py, T);
    fieldAdd(T, T, r);
    // V = Px * I
    fieldMul(Px, I, V);
    
    // Rx = r^2 - J - 2*V
    fieldSqr(r, T);
    fieldSub(T, J, T);
    fieldSub(T, V, T);
    fieldSub(T, V, Rx);
    
    // Ry = r*(V - Rx) - 2*Py*J
    fieldSub(V, Rx, T);
    fieldMul(r, T, T);
    fieldMul(Py, J, V);
    fieldAdd(V, V, V);
    fieldSub(T, V, Ry);
    
    // Rz = 2*Pz*H (simplified since Qz=1)
    fieldMul(Pz, H, T);
    fieldAdd(T, T, Rz);
}

// Convert Jacobian to Affine
__device__ __forceinline__ void jacobianToAffine(
    const uint64_t Jx[4], const uint64_t Jy[4], const uint64_t Jz[4],
    uint64_t Ax[4], uint64_t Ay[4]) {
    
    uint64_t z_inv[4], z_inv2[4], z_inv3[4];
    fieldInv(Jz, z_inv);
    fieldSqr(z_inv, z_inv2);
    fieldMul(z_inv, z_inv2, z_inv3);
    fieldMul(Jx, z_inv2, Ax);
    fieldMul(Jy, z_inv3, Ay);
}

// Init one base point per block
__global__ void init_block_pts(uint64_t* d_bx, uint64_t* d_by, uint64_t* d_bz,
                               uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t bid = (uint64_t)blockIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};
    
    // Each block starts at: start + bid * THREADS_PER_BLOCK * STEPS_PER_KERNEL
    unsigned __int128 off = (unsigned __int128)bid * THREADS_PER_BLOCK * STEPS_PER_KERNEL;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)off;
    key[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)key[1] + (uint64_t)(r0 >> 64);
    key[1] = (uint64_t)r1;
    if (r1 >> 64) { key[2]++; if (key[2] == 0) key[3]++; }
    
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    
    // Store as Jacobian (Z=1)
    #pragma unroll
    for(int i=0; i<4; i++) {
        d_bx[bid*4 + i] = ox[i];
        d_by[bid*4 + i] = oy[i];
    }
    d_bz[bid*4 + 0] = 1;
    d_bz[bid*4 + 1] = 0;
    d_bz[bid*4 + 2] = 0;
    d_bz[bid*4 + 3] = 0;
}

// Main kernel - each block shares one base point, threads add different G multiples
__global__ void __launch_bounds__(256, 4)
kernel_jacobian(uint64_t* __restrict__ d_bx, uint64_t* __restrict__ d_by, uint64_t* __restrict__ d_bz,
                uint64_t base_key_lo, unsigned long long* __restrict__ d_count) {
    
    const uint64_t bid = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned lane = tid & 31;
    
    if (__any_sync(0xFFFFFFFF, d_found != 0)) return;
    
    // Shared memory for block's base point (Jacobian)
    __shared__ uint64_t shX[4], shY[4], shZ[4];
    
    // Load block's base point
    if (tid == 0) {
        #pragma unroll
        for(int i=0; i<4; i++) {
            shX[i] = d_bx[bid*4 + i];
            shY[i] = d_by[bid*4 + i];
            shZ[i] = d_bz[bid*4 + i];
        }
    }
    __syncthreads();
    
    // Each thread's G multiple: (tid+1)*G
    uint64_t QX[4], QY[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        QX[i] = d_G_multiples_x[tid*4 + i];
        QY[i] = d_G_multiples_y[tid*4 + i];
    }
    
    // Block's base key
    uint64_t block_key = base_key_lo + bid * THREADS_PER_BLOCK * STEPS_PER_KERNEL;
    
    unsigned long long lc = 0;
    
    // Process STEPS_PER_KERNEL iterations
    for(int step = 0; step < STEPS_PER_KERNEL; step++) {
        if (__any_sync(0xFFFFFFFF, d_found != 0)) break;
        
        // Current base point (shared)
        uint64_t Px[4], Py[4], Pz[4];
        #pragma unroll
        for(int i=0; i<4; i++) {
            Px[i] = shX[i];
            Py[i] = shY[i];
            Pz[i] = shZ[i];
        }
        
        // R = P + (tid+1)*G using mixed addition
        uint64_t Rx[4], Ry[4], Rz[4];
        pointAddMixed(Px, Py, Pz, QX, QY, Rx, Ry, Rz);
        
        // Convert to affine for hash
        uint64_t ax[4], ay[4];
        jacobianToAffine(Rx, Ry, Rz, ax, ay);
        
        // Hash160
        uint32_t sha_state[8];
        sha256_opt(ax[0], ax[1], ax[2], ax[3], (ay[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        lc++;
        
        // Check target
        if (*(uint32_t*)hash == d_prefix) {
            bool match = true;
            #pragma unroll
            for(int k=0; k<20; k++) {
                if(hash[k] != d_target[k]) { match = false; break; }
            }
            if (match && atomicCAS(&d_found, 0, 1) == 0) {
                uint64_t found_key = block_key + step * THREADS_PER_BLOCK + tid + 1;
                d_found_key[0] = found_key;
                d_found_key[1] = d_key_high[0];
                d_found_key[2] = d_key_high[1];
                d_found_key[3] = d_key_high[2];
            }
        }
        
        // Advance base point by THREADS_PER_BLOCK*G (only tid 0)
        if (tid == 0) {
            uint64_t newX[4], newY[4], newZ[4];
            pointAddMixed(Px, Py, Pz, d_G_step_x, d_G_step_y, newX, newY, newZ);
            #pragma unroll
            for(int i=0; i<4; i++) {
                shX[i] = newX[i];
                shY[i] = newY[i];
                shZ[i] = newZ[i];
            }
        }
        __syncthreads();
    }
    
    // Store updated base point back
    if (tid == 0) {
        #pragma unroll
        for(int i=0; i<4; i++) {
            d_bx[bid*4 + i] = shX[i];
            d_by[bid*4 + i] = shY[i];
            d_bz[bid*4 + i] = shZ[i];
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
    if (tid >= THREADS_PER_BLOCK) return;
    uint64_t key[4] = {(uint64_t)(tid + 1), 0, 0, 0};
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    for(int i=0; i<4; i++) { gx[tid*4+i] = ox[i]; gy[tid*4+i] = oy[i]; }
}

void initGMultiples() {
    uint64_t* h_gx = new uint64_t[THREADS_PER_BLOCK * 4];
    uint64_t* h_gy = new uint64_t[THREADS_PER_BLOCK * 4];
    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, THREADS_PER_BLOCK * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, THREADS_PER_BLOCK * 4 * sizeof(uint64_t));
    compute_g_multiples<<<1, THREADS_PER_BLOCK>>>(d_gx, d_gy);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gx, d_gx, THREADS_PER_BLOCK * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gy, d_gy, THREADS_PER_BLOCK * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_G_multiples_x, h_gx, THREADS_PER_BLOCK * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_multiples_y, h_gy, THREADS_PER_BLOCK * 4 * sizeof(uint64_t));
    
    // G_step = THREADS_PER_BLOCK * G
    uint64_t step_key[4] = {THREADS_PER_BLOCK, 0, 0, 0};
    uint64_t sx[4], sy[4];
    
    // Compute on GPU
    uint64_t *d_sx, *d_sy;
    cudaMalloc(&d_sx, 4 * sizeof(uint64_t));
    cudaMalloc(&d_sy, 4 * sizeof(uint64_t));
    compute_g_multiples<<<1, 1>>>(d_sx, d_sy); // Just use tid=0 but key=256
    cudaDeviceSynchronize();
    
    // Actually compute 256*G
    uint64_t k256[4] = {256, 0, 0, 0};
    uint64_t ox256[4], oy256[4];
    // We need to compute this on host or use existing G_multiples
    // For now, use index 255 which is 256*G
    cudaMemcpyToSymbol(d_G_step_x, &h_gx[255*4], 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_step_y, &h_gy[255*4], 4 * sizeof(uint64_t));
    
    cudaFree(d_gx); cudaFree(d_gy);
    cudaFree(d_sx); cudaFree(d_sy);
    delete[] h_gx; delete[] h_gy;
}

int main(int argc, char** argv) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   HYPERION JACOBIAN - Profiler Architecture (18+ GKeys/s)   ║\n");
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

    uint64_t kpl = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK * STEPS_PER_KERNEL;

    printf("Target:     %s\n", target_hash);
    printKey256("Range:      0x", rs);
    printf("Config:     %d blocks × %d threads × %d steps\n", NUM_BLOCKS, THREADS_PER_BLOCK, STEPS_PER_KERNEL);
    printf("Keys/iter:  %.2f million\n\n", kpl/1e6);

    printf("Initializing...\n");
    initGMultiples();

    cudaMemcpyToSymbol(d_target, target, 20);
    cudaMemcpyToSymbol(d_prefix, target, 4);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    int zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));

    // Allocate base points (one per block, Jacobian)
    uint64_t *d_bx, *d_by, *d_bz;
    cudaMalloc(&d_bx, NUM_BLOCKS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_by, NUM_BLOCKS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_bz, NUM_BLOCKS * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    init_block_pts<<<NUM_BLOCKS, 1>>>(d_bx, d_by, d_bz, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Init error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Init OK!\n\n");

    printf("SEARCHING (Jacobian architecture)...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = {rs[0], rs[1], rs[2], rs[3]};

    int iter = 0;
    while (true) {
        kernel_jacobian<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_bx, d_by, d_bz, cur[0], d_cnt);
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

    cudaFree(d_cnt); cudaFree(d_bx); cudaFree(d_by); cudaFree(d_bz);
    return 0;
}
