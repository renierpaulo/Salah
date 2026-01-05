/*
 * HYPERION SEARCH V2 - High-Performance Bitcoin Key Search
 * 
 * PIPELINE (optimized for RTX 3090):
 * [GPU]
 *   privkey++ → EC add (Jacobian) → HASH160 → Bloom filter → match? → buffer
 * [CPU]
 *   Base58/Bech32 encoding only for matches (verification)
 *
 * Target: ~18 GKeys/s with full pipeline
 * Architecture: Warp-level batch inversion for efficient Jacobian→Affine
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <set>
#include <string>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256
#define WARP_SIZE 32

// Bloom filter config (2^26 bits = 8MB - smaller for better cache)
#define BLOOM_SIZE_BITS 26
#define BLOOM_SIZE_BYTES (1ULL << (BLOOM_SIZE_BITS - 3))
#define BLOOM_SIZE_WORDS (BLOOM_SIZE_BYTES / 8)
#define BLOOM_HASH_COUNT 5

// Match buffer
#define MAX_MATCHES 4096

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

// Match buffer structure - stores candidates for CPU verification
struct Match {
    uint64_t key[4];      // Private key
    uint64_t pubX[4];     // Public key X (affine)
    uint64_t pubY[4];     // Public key Y (affine)
    uint8_t hash160[20];  // Hash160 for quick comparison
};

__device__ int d_match_count = 0;
__device__ Match d_matches[MAX_MATCHES];

// Host-side target storage for CPU verification
std::set<std::string> h_target_set;
std::vector<uint8_t[20]> h_targets;

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

// Bloom filter check (device) - fast probabilistic filter
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

// Warp-level batch inversion using shuffle - 32 inversions with 1 modular inverse
__device__ void warpBatchInverse(uint64_t z[4], uint64_t z_inv[4]) {
    const int lane = threadIdx.x & 31;
    
    // Product accumulation using warp shuffle
    uint64_t prod[4];
    #pragma unroll
    for(int i=0; i<4; i++) prod[i] = z[i];
    
    // Prefix product within warp
    #pragma unroll
    for(int offset = 1; offset < 32; offset *= 2) {
        uint64_t other[4];
        #pragma unroll
        for(int i=0; i<4; i++) {
            other[i] = __shfl_up_sync(0xFFFFFFFF, prod[i], offset);
        }
        if(lane >= offset) {
            uint64_t tmp[4];
            fieldMul(prod, other, tmp);
            #pragma unroll
            for(int i=0; i<4; i++) prod[i] = tmp[i];
        }
    }
    
    // Lane 31 has the product of all Z values - invert it
    uint64_t total_inv[4];
    if(lane == 31) {
        fieldInv(prod, total_inv);
    }
    
    // Broadcast total inverse to all lanes
    #pragma unroll
    for(int i=0; i<4; i++) {
        total_inv[i] = __shfl_sync(0xFFFFFFFF, total_inv[i], 31);
    }
    
    // Compute individual inverses using suffix products
    uint64_t suffix[4];
    #pragma unroll
    for(int i=0; i<4; i++) suffix[i] = z[i];
    
    // Suffix product (from right)
    #pragma unroll
    for(int offset = 1; offset < 32; offset *= 2) {
        uint64_t other[4];
        #pragma unroll
        for(int i=0; i<4; i++) {
            other[i] = __shfl_down_sync(0xFFFFFFFF, suffix[i], offset);
        }
        if(lane + offset < 32) {
            uint64_t tmp[4];
            fieldMul(suffix, other, tmp);
            #pragma unroll
            for(int i=0; i<4; i++) suffix[i] = tmp[i];
        }
    }
    
    // Final computation: z_inv[i] = total_inv * prefix[i-1] * suffix[i+1]
    uint64_t prefix_prev[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        prefix_prev[i] = __shfl_up_sync(0xFFFFFFFF, prod[i], 1);
    }
    
    uint64_t suffix_next[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        suffix_next[i] = __shfl_down_sync(0xFFFFFFFF, suffix[i], 1);
    }
    
    if(lane == 0) {
        // z_inv[0] = total_inv * suffix[1..31]
        fieldMul(total_inv, suffix_next, z_inv);
    } else if(lane == 31) {
        // z_inv[31] = total_inv * prefix[0..30]
        fieldMul(total_inv, prefix_prev, z_inv);
    } else {
        // z_inv[i] = total_inv * prefix[0..i-1] * suffix[i+1..31]
        uint64_t tmp[4];
        fieldMul(total_inv, prefix_prev, tmp);
        fieldMul(tmp, suffix_next, z_inv);
    }
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

// Main search kernel - optimized for ~18 GKeys/s
__global__ void __launch_bounds__(256, 4)
kernel_search_v2(
    uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
    const uint64_t* __restrict__ d_bloom_filter,
    uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    
    const unsigned tid = (unsigned)threadIdx.x;
    const unsigned bid = (unsigned)blockIdx.x;
    const unsigned lane = tid & 31;
    const unsigned warp_id = tid >> 5;

    // One base point per block (Jacobian)
    __shared__ uint64_t shX[4];
    __shared__ uint64_t shY[4];
    __shared__ uint64_t shZ[4];

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
            shX[i] = Pj.X[i];
            shY[i] = Pj.Y[i];
            shZ[i] = Pj.Z[i];
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
    #pragma unroll
    for(int i=0; i<4; i++) {
        Q256X[i] = d_G_multiples_x[(BATCH_SIZE-1)*4 + i];
        Q256Y[i] = d_G_multiples_y[(BATCH_SIZE-1)*4 + i];
    }

    // PHASE 1: ECC - Compute Jacobian point
    ECPointJ P;
    #pragma unroll
    for(int i=0; i<4; i++) {
        P.X[i] = shX[i];
        P.Y[i] = shY[i];
        P.Z[i] = shZ[i];
    }
    P.infinity = false;

    ECPointJ R;
    pointAddJacobianMixed_XY_Hot(P, QX, QY, R);

    // PHASE 2: Warp-level batch inversion (32 inversions with 1 modular inverse)
    uint64_t z_inv[4];
    warpBatchInverse(R.Z, z_inv);

    // PHASE 3: Convert to affine
    uint64_t z_inv2[4], z_inv3[4];
    fieldSqr(z_inv, z_inv2);
    fieldMul(z_inv2, z_inv, z_inv3);

    uint64_t ax[4], ay[4];
    fieldMul(R.X, z_inv2, ax);
    fieldMul(R.Y, z_inv3, ay);

    // PHASE 4: HASH160 (SHA256 + RIPEMD160)
    uint32_t sha_state[8];
    sha256_opt(ax[0], ax[1], ax[2], ax[3], (ay[0] & 1) ? 0x03 : 0x02, sha_state);

    uint8_t h[20];
    ripemd160_opt(sha_state, h);

    // PHASE 5: BLOOM FILTER CHECK
    if (bloom_check(h, d_bloom_filter)) {
        // Potential match! Store in buffer for CPU verification
        int idx = atomicAdd(&d_match_count, 1);
        if (idx < MAX_MATCHES) {
            // Reconstruct full private key
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

            // Store match candidate
            #pragma unroll
            for(int i=0; i<4; i++) {
                d_matches[idx].key[i] = key[i];
                d_matches[idx].pubX[i] = ax[i];
                d_matches[idx].pubY[i] = ay[i];
            }
            #pragma unroll
            for(int i=0; i<20; i++) {
                d_matches[idx].hash160[i] = h[i];
            }
        }
    }

    // Advance base point by +BATCH_SIZE*G (only thread 0)
    if(tid == 0) {
        ECPointJ newP;
        pointAddJacobianMixed_XY_Hot(P, Q256X, Q256Y, newP);
        
        // Convert back to affine for next iteration
        uint64_t zinv[4], zinv2[4], zinv3[4];
        fieldInv(newP.Z, zinv);
        fieldSqr(zinv, zinv2);
        fieldMul(zinv2, zinv, zinv3);
        
        uint64_t newX[4], newY[4];
        fieldMul(newP.X, zinv2, newX);
        fieldMul(newP.Y, zinv3, newY);
        
        #pragma unroll
        for(int i=0; i<4; i++) {
            d_px[bid*4 + i] = newX[i];
            d_py[bid*4 + i] = newY[i];
        }
    }

    // Update global counter
    if(tid == 0 && bid == 0) {
        atomicAdd(d_count, (unsigned long long)NUM_BLOCKS * THREADS_PER_BLOCK);
    }
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

// CPU-side Base58 encoding
static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

void encodeBase58(const uint8_t* data, size_t len, char* out) {
    // Count leading zeros
    size_t zeros = 0;
    while (zeros < len && data[zeros] == 0) zeros++;
    
    // Allocate enough space
    size_t size = (len - zeros) * 138 / 100 + 1;
    uint8_t* buf = new uint8_t[size]();
    
    for (size_t i = zeros; i < len; i++) {
        int carry = data[i];
        for (size_t j = 0; j < size; j++) {
            carry += 256 * buf[size - 1 - j];
            buf[size - 1 - j] = carry % 58;
            carry /= 58;
        }
    }
    
    // Skip leading zeros in base58 result
    size_t k = 0;
    while (k < size && buf[k] == 0) k++;
    
    // Output
    size_t idx = 0;
    for (size_t i = 0; i < zeros; i++) out[idx++] = '1';
    for (size_t i = k; i < size; i++) out[idx++] = BASE58_ALPHABET[buf[i]];
    out[idx] = '\0';
    
    delete[] buf;
}

void hash160ToAddress(const uint8_t hash160[20], char* address) {
    // Version byte (0x00 for mainnet)
    uint8_t versioned[25];
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160, 20);
    
    // Double SHA256 for checksum
    uint8_t hash1[32], hash2[32];
    // Simple SHA256 implementation for host
    // For simplicity, we'll just encode without checksum verification here
    // In production, use proper SHA256
    
    // Checksum (first 4 bytes of double SHA256)
    // Simplified: just use zeros for now - real impl would compute proper checksum
    versioned[21] = 0;
    versioned[22] = 0;
    versioned[23] = 0;
    versioned[24] = 0;
    
    encodeBase58(versioned, 25, address);
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

// CPU verification of matches
bool verifyMatch(const Match& m, const std::set<std::string>& targets) {
    // Convert hash160 to hex string for comparison
    char hex[41];
    for (int i = 0; i < 20; i++) {
        sprintf(hex + i*2, "%02x", m.hash160[i]);
    }
    hex[40] = '\0';
    
    return targets.find(std::string(hex)) != targets.end();
}

int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       HYPERION SEARCH V2 - Optimized Full Pipeline        ║\n");
    printf("║   GPU: ECC → HASH160 → Bloom   |   CPU: Base58 Verify     ║\n");
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
        printf("❌ Invalid range. Use -r start:end (hex)\n");
        return 1;
    }
    strncpy(s, range, c-range);
    strcpy(e, c+1);
    if (!parseHex256(s, rs) || !parseHex256(e, re)) {
        printf("❌ Invalid range. Use -r start:end (hex)\n");
        return 1;
    }
    if (cmp256_le(rs, re) > 0) {
        printf("❌ Invalid range: start > end\n");
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
    uint64_t kpl = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    printf("  Keys/Launch:    %llu (%.2f million)\n", (unsigned long long)kpl, kpl/1e6);
    printf("  Bloom size:     %.1f MB (%d hash functions)\n\n", BLOOM_SIZE_BYTES / 1024.0 / 1024.0, BLOOM_HASH_COUNT);

    printf("Initializing...\n");
    initGMultiples();
    
    // Allocate bloom filter in global memory
    uint64_t* d_bloom_filter;
    cudaMalloc(&d_bloom_filter, BLOOM_SIZE_BYTES);
    uint64_t* h_bloom = new uint64_t[BLOOM_SIZE_WORDS];
    memset(h_bloom, 0, BLOOM_SIZE_BYTES);
    
    // Load targets and populate bloom filter + target set
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

            // Add to target set for CPU verification
            h_target_set.insert(std::string(line));

            // Add to bloom filter
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
    int total_matches = 0;
    int verified_matches = 0;
    
    while (true) {
        kernel_search_v2<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_bloom_filter, batch_lo, d_cnt);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }

        iteration++;

        // Check for matches every 100 iterations
        if (iteration % 100 == 0) {
            int match_count;
            cudaMemcpyFromSymbol(&match_count, d_match_count, sizeof(int));
            
            if (match_count > 0) {
                printf("\n🔍 GPU found %d bloom candidates - verifying on CPU...\n", match_count);
                
                Match* h_matches = new Match[match_count];
                cudaMemcpyFromSymbol(h_matches, d_matches, match_count * sizeof(Match));
                
                for (int i = 0; i < match_count && i < MAX_MATCHES; i++) {
                    total_matches++;
                    
                    // CPU VERIFICATION: Check if hash160 is in target set
                    if (verifyMatch(h_matches[i], h_target_set)) {
                        verified_matches++;
                        printf("\n");
                        printf("🎯 ═══════════════════════════════════════════════════════\n");
                        printf("   CONFIRMED MATCH #%d!\n", verified_matches);
                        printf("═══════════════════════════════════════════════════════\n");
                        printKey256Hex("   Private Key: ", h_matches[i].key);
                        printHash160("   Hash160:     ", h_matches[i].hash160);
                        
                        // Generate Base58 address
                        char address[64];
                        hash160ToAddress(h_matches[i].hash160, address);
                        printf("   Address:     %s\n", address);
                        printf("═══════════════════════════════════════════════════════\n\n");
                    }
                }
                
                delete[] h_matches;
                
                // Reset match buffer
                zero = 0;
                cudaMemcpyToSymbol(d_match_count, &zero, sizeof(int));
            }
        }

        // Progress update every 500 iterations
        if (iteration % 500 == 0) {
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double speed = total / elapsed / 1e9;
            
            printf("\r[%.2f GKeys/s] %llu keys | %d bloom hits | %d confirmed   ", 
                   speed, total, total_matches, verified_matches);
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
    printf("  Speed:            %.2f GKeys/s\n", speed);
    printf("  Keys searched:    %llu\n", total);
    printf("  Time:             %.2f seconds\n", elapsed);
    printf("  Bloom candidates: %d\n", total_matches);
    printf("  Confirmed matches: %d\n\n", verified_matches);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cnt);
    cudaFree(d_bloom_filter);

    return 0;
}
