/*
 * HYPERION HALF HASH - Early Exit Optimization
 * 
 * RADICAL APPROACH: SHA256 has 64 rounds. The first 32 rounds already produce
 * significant mixing. We can use partial SHA256 state as a FILTER:
 * 
 * 1. Compute first 32 rounds of SHA256 (half cost)
 * 2. Check if intermediate state has specific patterns unlikely in target
 * 3. Only compute full hash + RIPEMD160 for ~1% of candidates
 * 
 * This could give 10-50x speedup if filter is effective!
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"

// SHA256 Constants
__device__ __constant__ uint32_t K_SHA[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define S0(x) (ROR(x, 2) ^ ROR(x, 13) ^ ROR(x, 22))
#define S1(x) (ROR(x, 6) ^ ROR(x, 11) ^ ROR(x, 25))
#define s0(x) (ROR(x, 7) ^ ROR(x, 18) ^ ((x) >> 3))
#define s1(x) (ROR(x, 17) ^ ROR(x, 19) ^ ((x) >> 10))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

// Partial SHA256 - first 32 rounds only (for filtering)
__device__ __forceinline__ uint32_t sha256_partial_32rounds(
    uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, uint8_t parity) {
    
    uint32_t w[32];
    w[0] = ((uint32_t)parity << 24) | ((x3 >> 40) & 0xFFFFFF);
    w[1] = (uint32_t)(x3 >> 8);
    w[2] = ((uint32_t)(x3 & 0xFF) << 24) | ((x2 >> 40) & 0xFFFFFF);
    w[3] = (uint32_t)(x2 >> 8);
    w[4] = ((uint32_t)(x2 & 0xFF) << 24) | ((x1 >> 40) & 0xFFFFFF);
    w[5] = (uint32_t)(x1 >> 8);
    w[6] = ((uint32_t)(x1 & 0xFF) << 24) | ((x0 >> 40) & 0xFFFFFF);
    w[7] = (uint32_t)(x0 >> 8);
    w[8] = ((uint32_t)(x0 & 0xFF) << 24) | 0x800000;
    for(int i=9; i<15; i++) w[i] = 0;
    w[15] = 264;
    
    #pragma unroll
    for(int i = 16; i < 32; i++) 
        w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];
    
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    
    #pragma unroll
    for(int i = 0; i < 32; i++) {
        uint32_t t1 = h[7] + S1(h[4]) + CH(h[4], h[5], h[6]) + K_SHA[i] + w[i];
        uint32_t t2 = S0(h[0]) + MAJ(h[0], h[1], h[2]);
        h[7]=h[6]; h[6]=h[5]; h[5]=h[4]; h[4]=h[3]+t1; h[3]=h[2]; h[2]=h[1]; h[1]=h[0]; h[0]=t1+t2;
    }
    
    // Return a hash of the intermediate state for filtering
    return h[0] ^ h[4]; // XOR of two state words as filter value
}

// Full SHA256
__device__ __forceinline__ void sha256_full(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, 
                                             uint8_t parity, uint32_t* out) {
    uint32_t w[64];
    w[0] = ((uint32_t)parity << 24) | ((x3 >> 40) & 0xFFFFFF);
    w[1] = (uint32_t)(x3 >> 8);
    w[2] = ((uint32_t)(x3 & 0xFF) << 24) | ((x2 >> 40) & 0xFFFFFF);
    w[3] = (uint32_t)(x2 >> 8);
    w[4] = ((uint32_t)(x2 & 0xFF) << 24) | ((x1 >> 40) & 0xFFFFFF);
    w[5] = (uint32_t)(x1 >> 8);
    w[6] = ((uint32_t)(x1 & 0xFF) << 24) | ((x0 >> 40) & 0xFFFFFF);
    w[7] = (uint32_t)(x0 >> 8);
    w[8] = ((uint32_t)(x0 & 0xFF) << 24) | 0x800000;
    for(int i=9; i<15; i++) w[i] = 0;
    w[15] = 264;
    #pragma unroll
    for(int i = 16; i < 64; i++) w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];
    
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    
    #pragma unroll
    for(int i = 0; i < 64; i++) {
        uint32_t t1 = h[7] + S1(h[4]) + CH(h[4], h[5], h[6]) + K_SHA[i] + w[i];
        uint32_t t2 = S0(h[0]) + MAJ(h[0], h[1], h[2]);
        h[7]=h[6]; h[6]=h[5]; h[5]=h[4]; h[4]=h[3]+t1; h[3]=h[2]; h[2]=h[1]; h[1]=h[0]; h[0]=t1+t2;
    }
    
    out[0]=0x6a09e667+h[0]; out[1]=0xbb67ae85+h[1]; out[2]=0x3c6ef372+h[2]; out[3]=0xa54ff53a+h[3];
    out[4]=0x510e527f+h[4]; out[5]=0x9b05688c+h[5]; out[6]=0x1f83d9ab+h[6]; out[7]=0x5be0cd19+h[7];
}

// RIPEMD160
#define ROTL32(x,n) (((x)<<(n))|((x)>>(32-(n))))
#define F1(x,y,z) ((x)^(y)^(z))
#define F2(x,y,z) (((x)&(y))|(~(x)&(z)))
#define F3(x,y,z) (((x)|~(y))^(z))
#define F4(x,y,z) (((x)&(z))|((y)&~(z)))
#define F5(x,y,z) ((x)^((y)|~(z)))

__device__ __forceinline__ void ripemd160_full(const uint32_t* msg, uint8_t* out) {
    uint32_t x[16];
    #pragma unroll
    for(int i=0; i<8; i++) { 
        uint32_t v = msg[i]; 
        x[i] = ((v&0xFF)<<24) | ((v&0xFF00)<<8) | ((v>>8)&0xFF00) | (v>>24); 
    }
    x[8]=0x80; for(int i=9; i<14; i++) x[i]=0; x[14]=256; x[15]=0;
    
    uint32_t al=0x67452301,bl=0xefcdab89,cl=0x98badcfe,dl=0x10325476,el=0xc3d2e1f0;
    uint32_t ar=al,br=bl,cr=cl,dr=dl,er=el;
    
    const int rl[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int sl[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const int rr[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const int sr[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    
    #pragma unroll
    for(int j=0;j<80;j++){uint32_t f,k;if(j<16){f=F1(bl,cl,dl);k=0;}else if(j<32){f=F2(bl,cl,dl);k=0x5a827999;}else if(j<48){f=F3(bl,cl,dl);k=0x6ed9eba1;}else if(j<64){f=F4(bl,cl,dl);k=0x8f1bbcdc;}else{f=F5(bl,cl,dl);k=0xa953fd4e;}uint32_t t=ROTL32(al+f+x[rl[j]]+k,sl[j])+el;al=el;el=dl;dl=ROTL32(cl,10);cl=bl;bl=t;}
    #pragma unroll
    for(int j=0;j<80;j++){uint32_t f,k;if(j<16){f=F5(br,cr,dr);k=0x50a28be6;}else if(j<32){f=F4(br,cr,dr);k=0x5c4dd124;}else if(j<48){f=F3(br,cr,dr);k=0x6d703ef3;}else if(j<64){f=F2(br,cr,dr);k=0x7a6d76e9;}else{f=F1(br,cr,dr);k=0;}uint32_t t=ROTL32(ar+f+x[rr[j]]+k,sr[j])+er;ar=er;er=dr;dr=ROTL32(cr,10);cr=br;br=t;}
    
    uint32_t t=0xefcdab89U+cl+dr,h1=0x98badcfeU+dl+er,h2=0x10325476U+el+ar,h3=0xc3d2e1f0U+al+br,h4=0x67452301U+bl+cr,h0=t;
    out[0]=h0;out[1]=h0>>8;out[2]=h0>>16;out[3]=h0>>24;
    out[4]=h1;out[5]=h1>>8;out[6]=h1>>16;out[7]=h1>>24;
    out[8]=h2;out[9]=h2>>8;out[10]=h2>>16;out[11]=h2>>24;
    out[12]=h3;out[13]=h3>>8;out[14]=h3>>16;out[15]=h3>>24;
    out[16]=h4;out[17]=h4>>8;out[18]=h4>>16;out[19]=h4>>24;
}

__device__ __constant__ uint64_t d_key_high[3];
__device__ __constant__ uint64_t d_G_multiples_x[BATCH_SIZE * 4];
__device__ __constant__ uint64_t d_G_multiples_y[BATCH_SIZE * 4];
__device__ __constant__ uint8_t d_target[20];
__device__ __constant__ uint32_t d_prefix;
__device__ __constant__ uint32_t d_filter_value; // Pre-computed filter for target

__device__ int d_found = 0;
__device__ uint64_t d_found_key[4];
__device__ unsigned long long d_filter_hits = 0;

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

__global__ void init_pts(uint64_t* d_px, uint64_t* d_py, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};
    
    unsigned __int128 total_off = (unsigned __int128)tid * BATCH_SIZE;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)total_off;
    key[0] = (uint64_t)r0;
    unsigned __int128 r1 = (unsigned __int128)key[1] + (uint64_t)(r0 >> 64);
    key[1] = (uint64_t)r1;
    if (r1 >> 64) { key[2]++; if (key[2] == 0) key[3]++; }
    
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    
    #pragma unroll
    for(int i=0; i<4; i++) {
        d_px[tid*4 + i] = ox[i];
        d_py[tid*4 + i] = oy[i];
    }
}

// HALF-HASH KERNEL - Uses partial SHA256 as filter
__global__ void __launch_bounds__(256, 4)
kernel_halfhash(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py, 
                uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31;
    
    if (__any_sync(0xFFFFFFFF, d_found != 0)) return;

    uint64_t base_x[4], base_y[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        base_x[i] = d_px[tid*4 + i];
        base_y[i] = d_py[tid*4 + i];
    }

    uint64_t local_key = start_key_lo + (tid * BATCH_SIZE);
    unsigned long long lc = 0;

    // Batch inversion
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

    uint64_t inv[4];
    fieldInv_Fermat(products[BATCH_SIZE-1], inv);
    uint64_t inv_dx_last[4];

    #pragma unroll 16
    for(int i=BATCH_SIZE-1; i>=0; i--) {
        if (__any_sync(0xFFFFFFFF, d_found != 0)) break;

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

        uint64_t dy[4];
        fieldSub(&d_G_multiples_y[i*4], base_y, dy);
        
        uint64_t lambda[4];
        fieldMul(dy, inv_dx, lambda);
        
        uint64_t lambda_sq[4], temp[4], x3[4];
        fieldSqr(lambda, lambda_sq);
        fieldSub(lambda_sq, base_x, temp);
        fieldSub(temp, &d_G_multiples_x[i*4], x3);
        
        uint64_t y3[4];
        fieldSub(base_x, x3, temp);
        fieldMul(lambda, temp, y3);
        fieldSub(y3, base_y, y3);

        lc++;
        uint8_t par = (y3[0] & 1) ? 0x03 : 0x02;

        // EARLY EXIT: Use partial SHA256 as filter (skip ~99% of full hashes)
        // For now, just do full hash (filter disabled for correctness)
        uint32_t sha[8];
        sha256_full(x3[0], x3[1], x3[2], x3[3], par, sha);
        
        uint8_t hash[20];
        ripemd160_full(sha, hash);
        
        if(*(uint32_t*)hash == d_prefix) {
            bool m = true;
            for(int k=0; k<20; k++) if(hash[k]!=d_target[k]){m=false;break;}
            if(m && atomicCAS(&d_found,0,1)==0) {
                d_found_key[0]=local_key+i+1;
                d_found_key[1]=d_key_high[0];
                d_found_key[2]=d_key_high[1];
                d_found_key[3]=d_key_high[2];
            }
        }

        uint64_t tmp[4];
        fieldMul(inv, dx, tmp);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp[j];
    }

    // Update base point
    uint64_t dx_last[4], dy_last[4], lam[4], lam_sq[4], t[4];
    fieldSub(&d_G_multiples_x[(BATCH_SIZE-1)*4], base_x, dx_last);
    fieldSub(&d_G_multiples_y[(BATCH_SIZE-1)*4], base_y, dy_last);
    fieldMul(dy_last, inv_dx_last, lam);
    fieldSqr(lam, lam_sq);
    fieldSub(lam_sq, base_x, t);
    fieldSub(t, &d_G_multiples_x[(BATCH_SIZE-1)*4], base_x);
    
    uint64_t old_x[4];
    #pragma unroll
    for(int j=0; j<4; j++) old_x[j] = d_px[tid*4 + j];
    
    fieldSub(old_x, base_x, t);
    fieldMul(lam, t, base_y);
    fieldSub(base_y, &d_py[tid*4], base_y);
    
    #pragma unroll
    for(int j=0; j<4; j++) {
        d_px[tid*4 + j] = base_x[j];
        d_py[tid*4 + j] = base_y[j];
    }
    
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if(lane == 0) atomicAdd(d_count, lc);
}

__global__ void compute_g_multiples(uint64_t* gx, uint64_t* gy) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BATCH_SIZE) return;
    uint64_t key[4] = {(uint64_t)(tid + 1), 0, 0, 0};
    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);
    for(int i=0; i<4; i++) { gx[tid*4+i] = ox[i]; gy[tid*4+i] = oy[i]; }
}

void initGMultiples() {
    uint64_t* h_gx = new uint64_t[BATCH_SIZE * 4];
    uint64_t* h_gy = new uint64_t[BATCH_SIZE * 4];
    uint64_t *d_gx, *d_gy;
    cudaMalloc(&d_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMalloc(&d_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    compute_g_multiples<<<1, BATCH_SIZE>>>(d_gx, d_gy);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gx, d_gx, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gy, d_gy, BATCH_SIZE * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_G_multiples_x, h_gx, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_G_multiples_y, h_gy, BATCH_SIZE * 4 * sizeof(uint64_t));
    cudaFree(d_gx); cudaFree(d_gy);
    delete[] h_gx; delete[] h_gy;
}

int main(int argc, char** argv) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   HYPERION HALF HASH - Early Exit Optimization              ║\n");
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

    uint64_t total_threads = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    uint64_t kpl = total_threads * BATCH_SIZE;

    printf("Target:     %s\n", target_hash);
    printKey256("Range:      0x", rs);
    printf("Config:     %d blocks × %d threads × %d batch\n", 
           NUM_BLOCKS, THREADS_PER_BLOCK, BATCH_SIZE);
    printf("Keys/iter:  %.2f million\n\n", kpl/1e6);

    printf("Initializing...\n");
    initGMultiples();

    cudaMemcpyToSymbol(d_target, target, 20);
    cudaMemcpyToSymbol(d_prefix, target, 4);
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);

    int zero = 0;
    cudaMemcpyToSymbol(d_found, &zero, sizeof(int));

    uint64_t *d_px, *d_py;
    cudaMalloc(&d_px, total_threads * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, total_threads * 4 * sizeof(uint64_t));

    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    init_pts<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Init error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Init OK!\n\n");

    printf("SEARCHING...\n");
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = {rs[0], rs[1], rs[2], rs[3]};
    uint64_t batch_lo = cur[0];

    int iter = 0;
    while (true) {
        kernel_halfhash<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_lo, d_cnt);
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
            break;
        }

        if (iter % 10 == 0) {
            unsigned long long total;
            cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            printf("\r[%.2f GKeys/s] %llu keys, iter %d   ", 
                   total / elapsed / 1e9, total, iter);
            fflush(stdout);
        }

        add_u64_to_256(cur, kpl);
        batch_lo = cur[0];
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
    
    printf("\n\n══════════════════════════════════════════════════════════════\n");
    printf("HALF HASH Final Stats:\n");
    printf("  Speed:        %.2f GKeys/s\n", total/elapsed/1e9);
    printf("  Total keys:   %llu\n", total);
    printf("  Time:         %.2fs\n", elapsed);
    printf("══════════════════════════════════════════════════════════════\n");

    cudaFree(d_cnt); cudaFree(d_px); cudaFree(d_py);
    return 0;
}
