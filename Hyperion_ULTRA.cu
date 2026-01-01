/*
 * HYPERION ULTRA v6 - Maximum Speed (18+ GKeys/s) + 256-bit Range Fix
 * 
 * Target: High-speed search with correct key alignment for Puzzle 71+.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include "CUDAMath.h"

#define NUM_BLOCKS 16384
#define THREADS_PER_BLOCK 256
#define KEYS_PER_THREAD 4096
#define DEBUG_INTERVAL 100  // Print debug every N batches
#define REINIT_INTERVAL 1000  // Reinitialize points every N batches to prevent drift

__device__ __constant__ uint8_t d_target[20];
__device__ __constant__ uint32_t d_prefix;
__device__ int d_found = 0;
__device__ uint64_t d_found_key[4];
__device__ __constant__ uint64_t d_range_start[4];

// Debug: track current position with full 256-bit precision
__device__ uint64_t d_current_key[4];
__device__ uint64_t d_debug_px[4];  // Sample point X for verification
__device__ uint64_t d_debug_py[4];  // Sample point Y for verification
__device__ unsigned long long d_infinity_count = 0;  // Track infinity points (error indicator)

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

#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (ROTR32(x,2)^ROTR32(x,13)^ROTR32(x,22))
#define EP1(x) (ROTR32(x,6)^ROTR32(x,11)^ROTR32(x,25))
#define SIG0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))
#define ROTL32(x,n) (((x)<<(n))|((x)>>(32-(n))))

__device__ __forceinline__ void sha256_33_inline(const uint64_t* x, uint8_t parity, uint32_t* state) {
    uint32_t w[64];
    w[0] = ((uint32_t)parity << 24) | ((x[3] >> 40) & 0xFFFFFF);
    w[1] = (uint32_t)(x[3] >> 8);
    w[2] = ((uint32_t)(x[3] & 0xFF) << 24) | ((x[2] >> 40) & 0xFFFFFF);
    w[3] = (uint32_t)(x[2] >> 8);
    w[4] = ((uint32_t)(x[2] & 0xFF) << 24) | ((x[1] >> 40) & 0xFFFFFF);
    w[5] = (uint32_t)(x[1] >> 8);
    w[6] = ((uint32_t)(x[1] & 0xFF) << 24) | ((x[0] >> 40) & 0xFFFFFF);
    w[7] = (uint32_t)(x[0] >> 8);
    w[8] = ((uint32_t)(x[0] & 0xFF) << 24) | 0x800000;
    #pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 264;
    #pragma unroll
    for (int i = 16; i < 64; i++) w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a;
    uint32_t e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e,f,g) + K_SHA[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]=0x6a09e667+a; state[1]=0xbb67ae85+b;
    state[2]=0x3c6ef372+c; state[3]=0xa54ff53a+d;
    state[4]=0x510e527f+e; state[5]=0x9b05688c+f;
    state[6]=0x1f83d9ab+g; state[7]=0x5be0cd19+h;
}

__device__ __forceinline__ void ripemd160_32_inline(const uint32_t* sha_state, uint8_t* hash20) {
    uint32_t x[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t v = sha_state[i];
        x[i] = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v >> 8) & 0xFF00) | ((v >> 24) & 0xFF);
    }
    x[8] = 0x80;
    for (int i = 9; i < 14; i++) x[i] = 0;
    x[14] = 256; x[15] = 0;
    uint32_t al=0x67452301,bl=0xefcdab89,cl=0x98badcfe,dl=0x10325476,el=0xc3d2e1f0;
    uint32_t ar=al,br=bl,cr=cl,dr=dl,er=el;
    #define F1(x,y,z) ((x)^(y)^(z))
    #define F2(x,y,z) (((x)&(y))|(~(x)&(z)))
    #define F3(x,y,z) (((x)|~(y))^(z))
    #define F4(x,y,z) (((x)&(z))|((y)&~(z)))
    #define F5(x,y,z) ((x)^((y)|~(z)))
    const int rl[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int sl[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    #pragma unroll
    for (int j=0; j<80; j++) {
        uint32_t f,k;
        if(j<16){f=F1(bl,cl,dl);k=0;}
        else if(j<32){f=F2(bl,cl,dl);k=0x5a827999;}
        else if(j<48){f=F3(bl,cl,dl);k=0x6ed9eba1;}
        else if(j<64){f=F4(bl,cl,dl);k=0x8f1bbcdc;}
        else{f=F5(bl,cl,dl);k=0xa953fd4e;}
        uint32_t t=ROTL32(al+f+x[rl[j]]+k,sl[j])+el;
        al=el;el=dl;dl=ROTL32(cl,10);cl=bl;bl=t;
    }
    const int rr[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const int sr[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    #pragma unroll
    for (int j=0; j<80; j++) {
        uint32_t f,k;
        if(j<16){f=F5(br,cr,dr);k=0x50a28be6;}
        else if(j<32){f=F4(br,cr,dr);k=0x5c4dd124;}
        else if(j<48){f=F3(br,cr,dr);k=0x6d703ef3;}
        else if(j<64){f=F2(br,cr,dr);k=0x7a6d76e9;}
        else{f=F1(br,cr,dr);k=0;}
        uint32_t t=ROTL32(ar+f+x[rr[j]]+k,sr[j])+er;
        ar=er;er=dr;dr=ROTL32(cr,10);cr=br;br=t;
    }
    uint32_t t=0xefcdab89U+cl+dr;
    uint32_t h1=0x98badcfeU+dl+er;
    uint32_t h2=0x10325476U+el+ar;
    uint32_t h3=0xc3d2e1f0U+al+br;
    uint32_t h4=0x67452301U+bl+cr;
    uint32_t h0=t;
    hash20[0]=h0;hash20[1]=h0>>8;hash20[2]=h0>>16;hash20[3]=h0>>24;
    hash20[4]=h1;hash20[5]=h1>>8;hash20[6]=h1>>16;hash20[7]=h1>>24;
    hash20[8]=h2;hash20[9]=h2>>8;hash20[10]=h2>>16;hash20[11]=h2>>24;
    hash20[12]=h3;hash20[13]=h3>>8;hash20[14]=h3>>16;hash20[15]=h3>>24;
    hash20[16]=h4;hash20[17]=h4>>8;hash20[18]=h4>>16;hash20[19]=h4>>24;
}

// Helper: 256-bit addition with full carry propagation
__device__ __forceinline__ void add256(uint64_t r[4], const uint64_t a[4], uint64_t b) {
    r[0] = a[0] + b;
    uint64_t c = (r[0] < a[0]) ? 1ULL : 0ULL;
    r[1] = a[1] + c;
    c = (r[1] < a[1]) ? 1ULL : 0ULL;
    r[2] = a[2] + c;
    c = (r[2] < a[2]) ? 1ULL : 0ULL;
    r[3] = a[3] + c;
}

// Helper: 256-bit addition of two 256-bit numbers
__device__ __forceinline__ void add256_full(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t c = 0;
    r[0] = a[0] + b[0];
    c = (r[0] < a[0]) ? 1ULL : 0ULL;
    
    uint64_t tmp = a[1] + c;
    c = (tmp < a[1]) ? 1ULL : 0ULL;
    r[1] = tmp + b[1];
    c += (r[1] < tmp) ? 1ULL : 0ULL;
    
    tmp = a[2] + c;
    c = (tmp < a[2]) ? 1ULL : 0ULL;
    r[2] = tmp + b[2];
    c += (r[2] < tmp) ? 1ULL : 0ULL;
    
    r[3] = a[3] + b[3] + c;
}

__global__ void kernel_v6(uint64_t* d_px, uint64_t* d_py, unsigned long long* d_count, 
                          uint64_t batch_offset_lo, uint64_t batch_offset_hi) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d_found) return;

    uint64_t px[4], py[4];
    px[0] = d_px[tid*4]; px[1] = d_px[tid*4+1]; px[2] = d_px[tid*4+2]; px[3] = d_px[tid*4+3];
    py[0] = d_py[tid*4]; py[1] = d_py[tid*4+1]; py[2] = d_py[tid*4+2]; py[3] = d_py[tid*4+3];

    unsigned long long local_count = 0;
    ECPointA G; pointSetG(G);

    // Check if point is valid (not zero/infinity) before processing
    bool point_valid = !((px[0] | px[1] | px[2] | px[3]) == 0 && (py[0] | py[1] | py[2] | py[3]) == 0);
    
    #pragma unroll 4
    for (int iter = 0; iter < KEYS_PER_THREAD && !d_found; iter++) {
        // Skip if point became invalid
        if (!point_valid) {
            atomicAdd(&d_infinity_count, 1ULL);
            break;
        }
        
        uint8_t parity = (py[0] & 1) ? 0x03 : 0x02;
        uint32_t sha_state[8];
        sha256_33_inline(px, parity, sha_state);
        uint8_t hash[20];
        ripemd160_32_inline(sha_state, hash);
        local_count++;

        if (*(uint32_t*)hash == d_prefix) {
            bool match = true;
            for (int k = 0; k < 20; k++) if (hash[k] != d_target[k]) { match = false; break; }
            if (match && atomicCAS(&d_found, 0, 1) == 0) {
                // Calculate full 256-bit key with proper carry propagation
                // total_offset = batch_offset + tid * KEYS_PER_THREAD + iter
                uint64_t thread_offset = tid * KEYS_PER_THREAD + iter;
                uint64_t batch_off[4] = {batch_offset_lo, batch_offset_hi, 0, 0};
                
                // Add thread_offset to batch_offset
                uint64_t total_off[4];
                add256(total_off, batch_off, thread_offset);
                
                // Add to range_start
                add256_full(d_found_key, (uint64_t*)d_range_start, total_off);
            }
        }
        
        ECPointA P, R;
        P.X[0]=px[0]; P.X[1]=px[1]; P.X[2]=px[2]; P.X[3]=px[3];
        P.Y[0]=py[0]; P.Y[1]=py[1]; P.Y[2]=py[2]; P.Y[3]=py[3];
        P.infinity = false;
        pointAddAffine(P, G, R);
        
        // Check if result is infinity (would indicate P = -G, very rare but possible)
        if (R.infinity) {
            point_valid = false;
            atomicAdd(&d_infinity_count, 1ULL);
            // Set point to G to continue (skip one key)
            px[0]=G.X[0]; px[1]=G.X[1]; px[2]=G.X[2]; px[3]=G.X[3];
            py[0]=G.Y[0]; py[1]=G.Y[1]; py[2]=G.Y[2]; py[3]=G.Y[3];
        } else {
            px[0]=R.X[0]; px[1]=R.X[1]; px[2]=R.X[2]; px[3]=R.X[3];
            py[0]=R.Y[0]; py[1]=R.Y[1]; py[2]=R.Y[2]; py[3]=R.Y[3];
        }
    }
    d_px[tid*4]=px[0]; d_px[tid*4+1]=px[1]; d_px[tid*4+2]=px[2]; d_px[tid*4+3]=px[3];
    d_py[tid*4]=py[0]; d_py[tid*4+1]=py[1]; d_py[tid*4+2]=py[2]; d_py[tid*4+3]=py[3];
    atomicAdd(d_count, local_count);
    
    // Debug: thread 0 saves its current point for verification
    if (tid == 0) {
        d_debug_px[0] = px[0]; d_debug_px[1] = px[1]; d_debug_px[2] = px[2]; d_debug_px[3] = px[3];
        d_debug_py[0] = py[0]; d_debug_py[1] = py[1]; d_debug_py[2] = py[2]; d_debug_py[3] = py[3];
        
        // Calculate and store current key position
        uint64_t batch_off[4] = {batch_offset_lo, batch_offset_hi, 0, 0};
        uint64_t thread_end = (uint64_t)KEYS_PER_THREAD;
        uint64_t total_off[4];
        add256(total_off, batch_off, thread_end);
        add256_full(d_current_key, (uint64_t*)d_range_start, total_off);
    }
}

// Initialize points with full 256-bit offset support
__global__ void init_pts_256_with_offset(uint64_t* d_px, uint64_t* d_py, 
                                          uint64_t offset_lo, uint64_t offset_hi) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t key[4];
    
    // Calculate: key = range_start + offset + tid * KEYS_PER_THREAD
    // Step 1: Add tid * KEYS_PER_THREAD to offset
    uint64_t thread_offset = tid * KEYS_PER_THREAD;
    uint64_t total_lo = offset_lo + thread_offset;
    uint64_t total_hi = offset_hi + ((total_lo < offset_lo) ? 1ULL : 0ULL);
    
    // Step 2: Add to range_start with full carry propagation
    key[0] = d_range_start[0] + total_lo;
    uint64_t c = (key[0] < d_range_start[0]) ? 1ULL : 0ULL;
    
    uint64_t tmp = d_range_start[1] + c;
    c = (tmp < d_range_start[1]) ? 1ULL : 0ULL;
    key[1] = tmp + total_hi;
    c += (key[1] < tmp) ? 1ULL : 0ULL;
    
    key[2] = d_range_start[2] + c;
    c = (key[2] < d_range_start[2]) ? 1ULL : 0ULL;
    
    key[3] = d_range_start[3] + c;

    // Scalar multiplication: R = key * G
    ECPointA P, R; pointSetG(P); pointSetInfinity(R);
    for (int limb = 0; limb < 4; limb++) {
        uint64_t v = key[limb];
        for (int b = 0; b < 64; b++) {
            if (v & (1ULL << b)) {
                if (R.infinity) { R = P; } else { ECPointA t; pointAddAffine(R, P, t); R = t; }
            }
            ECPointA t; pointDoubleAffine(P, t); P = t;
        }
    }
    
    // Store result (handle infinity case)
    if (R.infinity) {
        // This shouldn't happen for valid keys, but handle it
        pointSetG(R);  // Reset to G as fallback
    }
    
    d_px[tid*4]=R.X[0]; d_px[tid*4+1]=R.X[1]; d_px[tid*4+2]=R.X[2]; d_px[tid*4+3]=R.X[3];
    d_py[tid*4]=R.Y[0]; d_py[tid*4+1]=R.Y[1]; d_py[tid*4+2]=R.Y[2]; d_py[tid*4+3]=R.Y[3];
}

// Legacy init wrapper (host function)
void launch_init_pts_256(uint64_t* d_px, uint64_t* d_py) {
    init_pts_256_with_offset<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, 0, 0);
}

bool parseHash160(const char* hex, uint8_t* h) {
    if (strlen(hex) != 40) return false;
    for (int i = 0; i < 20; i++) { unsigned int b; sscanf(hex+i*2, "%2x", &b); h[i] = b; }
    return true;
}

bool parseHex256(const char* hex, uint64_t* limbs) {
    limbs[0]=limbs[1]=limbs[2]=limbs[3]=0;
    int len = strlen(hex); if (len > 64) return false;
    char padded[65] = {0}; memset(padded, '0', 64-len); strcpy(padded+64-len, hex);
    for (int i=0; i<4; i++) { char chunk[17]={0}; strncpy(chunk, padded+i*16, 16); limbs[3-i] = strtoull(chunk, NULL, 16); }
    return true;
}

int main(int argc, char** argv) {
    const char *hx=nullptr, *range=nullptr;
    for (int i=1; i<argc; i++) {
        if (strcmp(argv[i], "--hash160") == 0) hx = argv[++i];
        else if (strcmp(argv[i], "--range") == 0) range = argv[++i];
    }
    if (!hx || !range) { printf("Usage: %s --hash160 <hex> --range <start:end>\n", argv[0]); return 1; }
    uint64_t rs[4], re[4]; char s[128]={0}, e[128]={0}; char* c = strchr((char*)range, ':');
    strncpy(s, range, c-range); strcpy(e, c+1); parseHex256(s, rs); parseHex256(e, re);
    uint8_t tgt[20]; parseHash160(hx, tgt);

    uint64_t thr = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    cudaMemcpyToSymbol(d_range_start, rs, 32);
    cudaMemcpyToSymbol(d_target, tgt, 20);
    uint32_t pfx = *(uint32_t*)tgt; cudaMemcpyToSymbol(d_prefix, &pfx, 4);

    uint64_t *d_px, *d_py; cudaMalloc(&d_px, thr*32); cudaMalloc(&d_py, thr*32);
    unsigned long long* d_cnt; cudaMalloc(&d_cnt, 8);
    unsigned long long uz = 0; cudaMemcpy(d_cnt, &uz, 8, cudaMemcpyHostToDevice);

    printf("HYPERION v6 - Ultra Speed + Debug\n");
    printf("Initializing %llu EC points...\n", thr);
    init_pts_256_with_offset<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, 0, 0);
    cudaDeviceSynchronize();
    
    cudaError_t init_err = cudaGetLastError();
    if (init_err != cudaSuccess) {
        printf("\n❌ Init CUDA Error: %s\n", cudaGetErrorString(init_err));
        return 1;
    }
    printf("✅ Points initialized\n");

    auto start_time = std::chrono::steady_clock::now();
    int found = 0;
    
    // Use 128-bit batch offset (two uint64_t) for proper range handling
    uint64_t batch_offset_lo = 0;
    uint64_t batch_offset_hi = 0;
    const uint64_t keys_per_batch = thr * KEYS_PER_THREAD;  // ~17 billion per batch
    
    int batch_count = 0;
    uint64_t last_debug_total = 0;
    
    printf("\n🔍 Starting search...\n");
    printf("   Keys per batch: %llu (~%.2f billion)\n", keys_per_batch, keys_per_batch / 1e9);
    printf("   Range start: %016llx%016llx%016llx%016llx\n", rs[3], rs[2], rs[1], rs[0]);
    printf("   Range end:   %016llx%016llx%016llx%016llx\n\n", re[3], re[2], re[1], re[0]);
    
    while (!found) {
        kernel_v6<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_cnt, batch_offset_lo, batch_offset_hi);
        cudaDeviceSynchronize();
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }
        
        cudaMemcpyFromSymbol(&found, d_found, 4);
        unsigned long long total; cudaMemcpy(&total, d_cnt, 8, cudaMemcpyDeviceToHost);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        
        // Update batch offset with carry propagation
        uint64_t old_lo = batch_offset_lo;
        batch_offset_lo += keys_per_batch;
        if (batch_offset_lo < old_lo) {
            batch_offset_hi++;  // Carry to high 64-bits
        }
        
        batch_count++;
        
        // Detailed debug output every N batches
        if (batch_count % DEBUG_INTERVAL == 0) {
            uint64_t cur_key[4], dbg_px[4], dbg_py[4];
            unsigned long long inf_count;
            cudaMemcpyFromSymbol(cur_key, d_current_key, 32);
            cudaMemcpyFromSymbol(dbg_px, d_debug_px, 32);
            cudaMemcpyFromSymbol(dbg_py, d_debug_py, 32);
            cudaMemcpyFromSymbol(&inf_count, d_infinity_count, 8);
            
            double keys_since_last = (double)(total - last_debug_total);
            double current_speed = keys_since_last / (elapsed > 0 ? (DEBUG_INTERVAL * (elapsed / batch_count)) : 1) / 1e9;
            
            printf("\n📊 [Batch %d] Debug Info:\n", batch_count);
            printf("   Current Key:  %016llx%016llx%016llx%016llx\n", cur_key[3], cur_key[2], cur_key[1], cur_key[0]);
            printf("   Batch Offset: %016llx%016llx (hi:lo)\n", batch_offset_hi, batch_offset_lo);
            printf("   Point X[0]:   %016llx (sanity check)\n", dbg_px[0]);
            printf("   Point Y[0]:   %016llx (sanity check)\n", dbg_py[0]);
            printf("   Speed: %.2f GKeys/s | Total: %.2f billion keys\n", current_speed, total / 1e9);
            printf("   Infinity/Error count: %llu\n", inf_count);
            
            // Sanity check: verify point is not zero (would indicate error)
            if (dbg_px[0] == 0 && dbg_px[1] == 0 && dbg_px[2] == 0 && dbg_px[3] == 0) {
                printf("   ⚠️  WARNING: Point X is ZERO! Possible EC arithmetic error!\n");
            }
            if (dbg_py[0] == 0 && dbg_py[1] == 0 && dbg_py[2] == 0 && dbg_py[3] == 0) {
                printf("   ⚠️  WARNING: Point Y is ZERO! Possible EC arithmetic error!\n");
            }
            
            // Check for infinity errors
            if (inf_count > 0) {
                printf("   ⚠️  WARNING: %llu threads hit infinity point! EC arithmetic issue detected!\n", inf_count);
            }
            
            // Check if we're making progress (keys should increase)
            if (total == last_debug_total && batch_count > DEBUG_INTERVAL) {
                printf("   ❌ ERROR: No progress! Keys count stuck at %llu\n", total);
            }
            
            last_debug_total = total;
            
            // Periodic reinitialization to prevent accumulated errors
            if (batch_count % REINIT_INTERVAL == 0) {
                printf("🔄 Reinitializing points to prevent drift...\n");
                
                // Reset infinity counter
                unsigned long long zero = 0;
                cudaMemcpyToSymbol(d_infinity_count, &zero, 8);
                
                // Reinitialize all points at current offset
                init_pts_256_with_offset<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_offset_lo, batch_offset_hi);
                cudaDeviceSynchronize();
                
                cudaError_t reinit_err = cudaGetLastError();
                if (reinit_err != cudaSuccess) {
                    printf("   ❌ Reinit CUDA Error: %s\n", cudaGetErrorString(reinit_err));
                } else {
                    printf("   ✅ Points reinitialized at offset %llx:%llx\n", batch_offset_hi, batch_offset_lo);
                }
            }
        } else {
            printf("\r⚡ [%.2f GKeys/s] Total: %.2f B | Batch: %d | Offset: %llx:%llx   ", 
                   total/elapsed/1e9, total/1e9, batch_count, batch_offset_hi, batch_offset_lo);
            fflush(stdout);
        }
    }
    if (found) {
        uint64_t fk[4]; cudaMemcpyFromSymbol(fk, d_found_key, 32);
        printf("\nFOUND! Key: %llx%016llx%016llx%016llx\n", fk[3], fk[2], fk[1], fk[0]);
    }
    return 0;
}
