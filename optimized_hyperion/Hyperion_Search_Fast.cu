/*
 * HYPERION SEARCH FAST - High-Performance Bitcoin Key Search with Batch Inversion
 * 
 * Pipeline: privkey++ → EC add (Jacobian) → Batch Inversion → HASH160 → Bloom → Match
 * Architecture: 1 block = 1 basepoint, batch inversion for all 256 points
 * Target: 10+ GKeys/s with full hash160 + bloom checking
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

// Bloom filter config (2^28 bits = 32MB)
#define BLOOM_SIZE_BITS 28
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)
#define BLOOM_HASH_COUNT 7

// Match buffer
#define MAX_MATCHES 1024

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

// Device constants
__device__ __constant__ uint64_t d_key_high[3];
__device__ __constant__ uint64_t d_G_multiples_x[BATCH_SIZE * 4];
__device__ __constant__ uint64_t d_G_multiples_y[BATCH_SIZE * 4];

// Match buffer
struct Match {
    uint64_t key[4];
    uint8_t hash160[20];
};

__device__ int d_match_count = 0;
__device__ Match d_matches[MAX_MATCHES];

// Helper functions
__device__ __host__ static __forceinline__ int cmp256_le(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ __host__ static __forceinline__ void add_u64_to_256(uint64_t a[4], uint64_t add) {
    unsigned __int128 r0 = (unsigned __int128)a[0] + (uint64_t)add;
    a[0] = (uint64_t)r0;
    unsigned __int128 carry = r0 >> 64;
    if (!carry) return;
    unsigned __int128 r1 = (unsigned __int128)a[1] + (uint64_t)carry;
    a[1] = (uint64_t)r1;
    carry = r1 >> 64;
    if (!carry) return;
    unsigned __int128 r2 = (unsigned __int128)a[2] + (uint64_t)carry;
    a[2] = (uint64_t)r2;
    carry = r2 >> 64;
    if (!carry) return;
    a[3] += (uint64_t)carry;
}

// Bloom filter check (device)
__device__ __forceinline__ bool bloom_check(const uint8_t hash160[20], const uint64_t* __restrict__ bloom_filter) {
    uint64_t seed = *(uint64_t*)hash160;
    uint64_t h1 = seed;
    uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
    
    #pragma unroll
    for (int i = 0; i < BLOOM_HASH_COUNT; i++) {
        uint64_t hash = h1 + i * h2;
        uint64_t bit_index = hash & ((1ULL << BLOOM_SIZE_BITS) - 1);
        uint64_t word_index = bit_index >> 6;
        uint64_t bit_offset = bit_index & 63;
        
        if (!(bloom_filter[word_index] & (1ULL << bit_offset))) {
            return false;
        }
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

__global__ void init_pts_blocks(uint64_t* outX, uint64_t* outY,
                                uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t bid = (uint64_t)blockIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};

    const unsigned __int128 off = (unsigned __int128)bid * (unsigned __int128)THREADS_PER_BLOCK * (unsigned __int128)BATCH_SIZE;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)off;
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
        outX[bid*4 + i] = ox[i];
        outY[bid*4 + i] = oy[i];
    }
}

__device__ __forceinline__ void pointAddJacobianG_Hot(const ECPointJ& P, ECPointJ& R);
__device__ __forceinline__ void pointAddJacobianMixed_XY_Hot(const ECPointJ& P, const uint64_t QX[4], const uint64_t QY[4], ECPointJ& R);

// Batch inversion using Montgomery trick
__device__ void batchInvBlock(uint64_t* z_array, uint64_t* inv_array, int count) {
    const int tid = threadIdx.x;
    
    // Shared memory for product tree
    __shared__ uint64_t products[BATCH_SIZE * 4];
    
    // Step 1: Build product tree (each thread computes cumulative product)
    if (tid < count) {
        uint64_t prod[4];
        #pragma unroll
        for(int i=0; i<4; i++) {
            prod[i] = z_array[tid*4 + i];
        }
        
        // Store initial value
        #pragma unroll
        for(int i=0; i<4; i++) {
            products[tid*4 + i] = prod[i];
        }
        
        // Accumulate products from previous threads
        for(int j = 1; j <= tid; j *= 2) {
            __syncthreads();
            if (tid >= j) {
                uint64_t prev[4];
                #pragma unroll
                for(int i=0; i<4; i++) {
                    prev[i] = products[(tid-j)*4 + i];
                }
                uint64_t temp[4];
                fieldMul(prod, prev, temp);
                #pragma unroll
                for(int i=0; i<4; i++) {
                    prod[i] = temp[i];
                }
            }
            __syncthreads();
            #pragma unroll
            for(int i=0; i<4; i++) {
                products[tid*4 + i] = prod[i];
            }
        }
    }
    __syncthreads();
    
    // Step 2: Invert the final product (only thread 0)
    __shared__ uint64_t global_inv[4];
    if (tid == 0 && count > 0) {
        uint64_t final_prod[4];
        #pragma unroll
        for(int i=0; i<4; i++) {
            final_prod[i] = products[(count-1)*4 + i];
        }
        uint64_t inv[4];
        fieldInv(final_prod, inv);
        #pragma unroll
        for(int i=0; i<4; i++) {
            global_inv[i] = inv[i];
        }
    }
    __syncthreads();
    
    // Step 3: Propagate inversions back (each thread computes its own inverse)
    if (tid < count) {
        uint64_t inv[4];
        #pragma unroll
        for(int i=0; i<4; i++) {
            inv[i] = global_inv[i];
        }
        
        // Multiply by inverses of elements after this one
        for(int j = count - 1; j > tid; j--) {
            uint64_t z_j[4];
            #pragma unroll
            for(int i=0; i<4; i++) {
                z_j[i] = z_array[j*4 + i];
            }
            uint64_t temp[4];
            fieldMul(inv, z_j, temp);
            #pragma unroll
            for(int i=0; i<4; i++) {
                inv[i] = temp[i];
            }
        }
        
        // Final inverse for this element
        if (tid > 0) {
            uint64_t prod_prev[4];
            #pragma unroll
            for(int i=0; i<4; i++) {
                prod_prev[i] = products[(tid-1)*4 + i];
            }
            uint64_t temp[4];
            fieldMul(inv, prod_prev, temp);
            #pragma unroll
            for(int i=0; i<4; i++) {
                inv_array[tid*4 + i] = temp[i];
            }
        } else {
            // Thread 0: compute 1/z_0 directly
            uint64_t z_0[4];
            #pragma unroll
            for(int i=0; i<4; i++) {
                z_0[i] = z_array[i];
            }
            uint64_t temp[4];
            fieldInv(z_0, temp);
            #pragma unroll
            for(int i=0; i<4; i++) {
                inv_array[i] = temp[i];
            }
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(256, 4)
kernel_search_fast(
    uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
    const uint64_t* __restrict__ d_bloom_filter,
    uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    
    const unsigned tid = (unsigned)threadIdx.x;
    const unsigned bid = (unsigned)blockIdx.x;

    // Shared memory for batch inversion
    __shared__ uint64_t shPX[BATCH_SIZE * 4];
    __shared__ uint64_t shPY[BATCH_SIZE * 4];
    __shared__ uint64_t shPZ[BATCH_SIZE * 4];
    __shared__ uint64_t shInv[BATCH_SIZE * 4];

    // One base point per block (Jacobian)
    __shared__ uint64_t shBaseX[4];
    __shared__ uint64_t shBaseY[4];
    __shared__ uint64_t shBaseZ[4];

    if(tid == 0) {
        uint64_t ax[4], ay[4];
        #pragma unroll
        for(int i=0; i<4; i++) {
            ax[i] = d_px[bid*4 + i];
            ay[i] = d_py[bid*4 + i];
        }

        ECPointA P_aff;
        #pragma unroll
        for(int i=0; i<4; i++) {
            P_aff.X[i] = ax[i];
            P_aff.Y[i] = ay[i];
        }
        P_aff.infinity = false;

        ECPointJ Pj;
        affineToJacobian(P_aff, Pj);

        #pragma unroll
        for(int i=0; i<4; i++) {
            shBaseX[i] = Pj.X[i];
            shBaseY[i] = Pj.Y[i];
            shBaseZ[i] = Pj.Z[i];
        }
    }
    __syncthreads();

    // Cache Q = (tid+1)*G (affine)
    uint64_t QX[4], QY[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        QX[i] = d_G_multiples_x[tid*4 + i];
        QY[i] = d_G_multiples_y[tid*4 + i];
    }

    // Cache Qstep = BATCH_SIZE*G
    uint64_t Q256X[4], Q256Y[4];
    if(tid == 0) {
        #pragma unroll
        for(int i=0; i<4; i++) {
            Q256X[i] = d_G_multiples_x[(BATCH_SIZE-1)*4 + i];
            Q256Y[i] = d_G_multiples_y[(BATCH_SIZE-1)*4 + i];
        }
    }

    // PHASE 1: ECC - Compute all 256 Jacobian points
    ECPointJ P;
    #pragma unroll
    for(int i=0; i<4; i++) {
        P.X[i] = shBaseX[i];
        P.Y[i] = shBaseY[i];
        P.Z[i] = shBaseZ[i];
    }
    P.infinity = false;

    ECPointJ R;
    pointAddJacobianMixed_XY_Hot(P, QX, QY, R);

    // Store Jacobian result in shared memory
    #pragma unroll
    for(int i=0; i<4; i++) {
        shPX[tid*4 + i] = R.X[i];
        shPY[tid*4 + i] = R.Y[i];
        shPZ[tid*4 + i] = R.Z[i];
    }

    // Advance base by +BATCH_SIZE*G for next iteration
    if(tid == 0) {
        ECPointJ newP;
        pointAddJacobianMixed_XY_Hot(P, Q256X, Q256Y, newP);
        #pragma unroll
        for(int i=0; i<4; i++) {
            shBaseX[i] = newP.X[i];
            shBaseY[i] = newP.Y[i];
            shBaseZ[i] = newP.Z[i];
        }
    }
    __syncthreads();

    // PHASE 2: BATCH INVERSION - Montgomery trick for all Z coordinates
    batchInvBlock(shPZ, shInv, BATCH_SIZE);
    __syncthreads();

    // PHASE 3: HASH & BLOOM - Convert to affine, hash, and check bloom
    uint64_t z_inv[4], z_inv2[4], z_inv3[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        z_inv[i] = shInv[tid*4 + i];
    }

    fieldSqr(z_inv, z_inv2);
    fieldMul(z_inv2, z_inv, z_inv3);

    uint64_t rx[4], ry[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        rx[i] = shPX[tid*4 + i];
        ry[i] = shPY[tid*4 + i];
    }

    uint64_t ax[4], ay[4];
    fieldMul(rx, z_inv2, ax);
    fieldMul(ry, z_inv3, ay);

    // HASH160: SHA256(compressed_pubkey) → RIPEMD160
    uint32_t sha_state[8];
    sha256_opt(ax[0], ax[1], ax[2], ax[3], (ay[0] & 1) ? 0x03 : 0x02, sha_state);

    uint8_t h[20];
    ripemd160_opt(sha_state, h);

    // BLOOM FILTER CHECK
    if (bloom_check(h, d_bloom_filter)) {
        // Potential match! Store in buffer
        int idx = atomicAdd(&d_match_count, 1);
        if (idx < MAX_MATCHES) {
            // Reconstruct full key for this point
            uint64_t key[4];
            key[0] = start_key_lo;
            key[1] = d_key_high[0];
            key[2] = d_key_high[1];
            key[3] = d_key_high[2];
            
            // Add block offset
            uint64_t block_offset = (uint64_t)bid * (uint64_t)THREADS_PER_BLOCK * (uint64_t)BATCH_SIZE;
            add_u64_to_256(key, block_offset);
            
            // Add thread offset
            add_u64_to_256(key, (uint64_t)tid);

            #pragma unroll
            for(int i=0; i<4; i++) d_matches[idx].key[i] = key[i];
            #pragma unroll
            for(int i=0; i<20; i++) d_matches[idx].hash160[i] = h[i];
        }
    }

    // Update global counter
    if(tid == 0 && bid == 0) {
        atomicAdd(d_count, (unsigned long long)NUM_BLOCKS * THREADS_PER_BLOCK * BATCH_SIZE);
    }

    // Store updated base back to global
    if(tid == 0) {
        ECPointJ Pj;
        #pragma unroll
        for(int i=0; i<4; i++) {
            Pj.X[i] = shBaseX[i];
            Pj.Y[i] = shBaseY[i];
            Pj.Z[i] = shBaseZ[i];
        }
        Pj.infinity = false;

        ECPointA Pa;
        jacobianToAffine(Pj, Pa);

        #pragma unroll
        for(int i=0; i<4; i++) {
            d_px[bid*4 + i] = Pa.X[i];
            d_py[bid*4 + i] = Pa.Y[i];
        }
    }
}

// Hot-path implementations
__device__ __forceinline__ void pointAddJacobianMixed_XY_Hot(const ECPointJ& P, const uint64_t QX[4], const uint64_t QY[4], ECPointJ& R) {
    uint64_t Z2[4], H[4], r[4], I[4], Jv[4], V[4];
    uint64_t T1[4], T2[4];

    fieldSqr(P.Z, Z2);
    fieldMul(QX, Z2, T1);

    fieldMul(P.Z, Z2, T2);
    fieldMul(QY, T2, T2);

    fieldSub(T1, P.X, H);
    fieldSub(T2, P.Y, r);
    fieldAdd(r, r, r);

    fieldAdd(H, H, I);
    fieldSqr(I, I);

    fieldMul(H, I, Jv);
    fieldMul(P.X, I, V);

    fieldSqr(r, R.X);
    fieldSub(R.X, Jv, R.X);
    fieldSub(R.X, V, R.X);
    fieldSub(R.X, V, R.X);

    fieldSub(V, R.X, T1);
    fieldMul(r, T1, R.Y);
    fieldMul(P.Y, Jv, T1);
    fieldAdd(T1, T1, T1);
    fieldSub(R.Y, T1, R.Y);

    fieldMul(P.Z, H, R.Z);
    fieldAdd(R.Z, R.Z, R.Z);

    R.infinity = false;
}

__device__ __forceinline__ void pointAddJacobianG_Hot(const ECPointJ& P, ECPointJ& R) {
    const uint64_t GX[4] = {0x79BE667EF9DCBBACULL, 0x55A06295CE870B07ULL, 0x029BFCDB2DCE28D9ULL, 0ULL};
    const uint64_t GY[4] = {0x483ADA7726A3C465ULL, 0x5DA4FBFC0E1108A8ULL, 0xFD17B448A6855419ULL, 0ULL};
    pointAddJacobianMixed_XY_Hot(P, GX, GY, R);
}

// Host functions
void initGMultiples() {
    uint64_t* h_gx = new uint64_t[BATCH_SIZE * 4];
    uint64_t* h_gy = new uint64_t[BATCH_SIZE * 4];

    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, BATCH_SIZE * 4 * sizeof(uint64_t));

    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;
    precompute_g_multiples_kernel<<<blocks, threads>>>(d_gx, d_gy);
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

int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║      HYPERION SEARCH FAST - Batch Inversion Pipeline      ║\n");
    printf("║    ECC → Batch Inv → HASH160 → Bloom → Match Detection    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    const char* targets_file = "targets.txt";
    const char* range = "1:ffffffffffffffffffff";

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-t") == 0 && (ai + 1) < argc) {
            targets_file = argv[++ai];
        } else if (strcmp(argv[ai], "-r") == 0 && (ai + 1) < argc) {
            range = argv[++ai];
        }
    }

    uint64_t rs[4], re[4];
    char s[128]={0}, e[128]={0};
    char* c = strchr((char*)range, ':');
    if (!c) {
        printf("❌ Range inválido. Use -r start:end (hex)\n");
        return 1;
    }
    strncpy(s, range, c-range);
    strcpy(e, c+1);
    if (!parseHex256(s, rs) || !parseHex256(e, re)) {
        printf("❌ Range inválido. Use -r start:end (hex)\n");
        return 1;
    }
    if (cmp256_le(rs, re) > 0) {
        printf("❌ Range inválido: start > end\n");
        return 1;
    }

    printf("Configuration:\n");
    printf("  Targets file:   %s\n", targets_file);
    {
        uint64_t tmp[4];
        for (int i=0;i<4;i++) tmp[i]=rs[i];
        printKey256("  Range start:    0x", tmp);
        for (int i=0;i<4;i++) tmp[i]=re[i];
        printKey256("  Range end:      0x", tmp);
    }
    printf("  Blocks:         %d\n", NUM_BLOCKS);
    printf("  Threads:        %d\n", THREADS_PER_BLOCK);
    printf("  Batch size:     %d\n", BATCH_SIZE);
    uint64_t kpl = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK * BATCH_SIZE;
    printf("  Keys/Launch:    %llu (%.2f million)\n", (unsigned long long)kpl, kpl/1e6);
    printf("  Bloom size:     %.1f MB (%d hash functions)\n\n", BLOOM_SIZE_BYTES / 1024.0 / 1024.0, BLOOM_HASH_COUNT);

    printf("Initializing...\n");
    initGMultiples();
    
    // Allocate and initialize bloom filter in global memory
    uint64_t* d_bloom_filter;
    cudaMalloc(&d_bloom_filter, BLOOM_SIZE_BYTES);
    uint64_t* h_bloom = new uint64_t[BLOOM_SIZE_WORDS];
    memset(h_bloom, 0, BLOOM_SIZE_BYTES);
    
    // Load targets and populate bloom filter
    FILE* f = fopen(targets_file, "r");
    if (!f) {
        printf("⚠️  Target file not found, using empty bloom filter\n");
    } else {
        char line[128];
        int count = 0;
        while (fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\r\n")] = 0;
            if (strlen(line) != 40) continue;

            uint8_t hash160[20];
            if (!parseHash160(line, hash160)) continue;

            uint64_t seed = *(uint64_t*)hash160;
            uint64_t h1 = seed;
            uint64_t h2 = seed * 0x9e3779b97f4a7c15ULL;
            
            for (int i = 0; i < BLOOM_HASH_COUNT; i++) {
                uint64_t hash = h1 + i * h2;
                uint64_t bit_index = hash & ((1ULL << BLOOM_SIZE_BITS) - 1);
                uint64_t word_index = bit_index >> 6;
                uint64_t bit_offset = bit_index & 63;
                h_bloom[word_index] |= (1ULL << bit_offset);
            }
            count++;
        }
        fclose(f);
        printf("✓ Loaded %d targets into bloom filter\n", count);
    }
    
    cudaMemcpy(d_bloom_filter, h_bloom, BLOOM_SIZE_BYTES, cudaMemcpyHostToDevice);
    delete[] h_bloom;

    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, (uint64_t)NUM_BLOCKS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, (uint64_t)NUM_BLOCKS * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    int zero = 0;
    cudaMemcpyToSymbol(d_match_count, &zero, sizeof(int));

    init_pts_blocks<<<NUM_BLOCKS, 1>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("❌ CUDA Error during init: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✓ Init OK!\n\n");

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                    SEARCHING...\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = { rs[0], rs[1], rs[2], rs[3] };
    uint64_t batch_lo = cur[0];

    int iteration = 0;
    while (true) {
        kernel_search_fast<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_bloom_filter, batch_lo, d_cnt);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }

        iteration++;

        // Check for matches every 10 iterations
        if (iteration % 10 == 0) {
            int match_count;
            cudaMemcpyFromSymbol(&match_count, d_match_count, sizeof(int));
            
            if (match_count > 0) {
                printf("\n🎯 Found %d potential match(es)!\n", match_count);
                
                Match* h_matches = new Match[match_count];
                cudaMemcpyFromSymbol(h_matches, d_matches, match_count * sizeof(Match));
                
                for (int i = 0; i < match_count && i < MAX_MATCHES; i++) {
                    printKey256("  Key:     0x", h_matches[i].key);
                    printHash160("  Hash160: ", h_matches[i].hash160);
                    printf("\n");
                }
                
                delete[] h_matches;
                
                // Reset match buffer
                zero = 0;
                cudaMemcpyToSymbol(d_match_count, &zero, sizeof(int));
            }
        }

        // Progress update every 50 iterations
        if (iteration % 50 == 0) {
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double speed = total / elapsed / 1e9;
            
            printf("\rIteration %d: [%.2f GKeys/s] %llu keys   ", iteration, speed, total);
            fflush(stdout);
        }

        // Advance range
        add_u64_to_256(cur, kpl);
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

    printf("\n\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                    SEARCH COMPLETE\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    printf("  Speed:          %.2f GKeys/s\n", speed);
    printf("  Keys searched:  %llu\n", total);
    printf("  Time:           %.2f seconds\n\n", elapsed);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cnt);
    cudaFree(d_bloom_filter);

    return 0;
}
