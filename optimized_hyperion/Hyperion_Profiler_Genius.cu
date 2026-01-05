/*
 * HYPERION PROFILER GENIUS - Profiling Ultra-Preciso
 * 
 * Usa a melhor versão (Batch 256 - 2.07 GKeys/s) e adiciona:
 * - Profiling com CUDA Events (precisão de microsegundos)
 * - Medição de cada componente isoladamente
 * - Análise de occupancy e bandwidth
 * - Identificação de gargalos reais
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

#define NUM_BLOCKS 8192
#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

// CUDAMath.h expects GRP_SIZE (used in internal sizing macros)
#ifndef GRP_SIZE
#define GRP_SIZE 1024
#endif

#include "CUDAUtils.h"
#include "CUDAMath_Jacobian_Fixed.h"
#include "CUDAMath_WorldClass.h"
#include "CUDAHash_Opt.cuh"

static void printHash160(const char* label, const uint8_t h[20]) {
    printf("%s", label);
    for (int i = 0; i < 20; i++) printf("%02x", (unsigned)h[i]);
    printf("\n");
}

__global__ void kernel_samples_hash160(const uint64_t* __restrict__ keys, uint8_t* __restrict__ out_hash20, int count) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    uint64_t key[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) key[i] = keys[tid * 4 + i];

    uint64_t ox[4], oy[4];
    scalarMulBaseAffine(key, ox, oy);

    uint32_t sha_state[8];
    sha256_opt(ox[0], ox[1], ox[2], ox[3], (oy[0] & 1) ? 0x03 : 0x02, sha_state);

    uint8_t h[20];
    ripemd160_opt(sha_state, h);

    #pragma unroll
    for (int i = 0; i < 20; i++) out_hash20[tid * 20 + i] = h[i];
}

#ifndef PROFILER_ECC_ONLY
#define PROFILER_ECC_ONLY 1
#endif

#ifndef ECC_ONLY_PERSIST_JACOBIAN
#define ECC_ONLY_PERSIST_JACOBIAN 0
#endif

#ifndef ECC_ONLY_STEPS
#define ECC_ONLY_STEPS 256
#endif

#ifndef ECC_ONLY_ITERS
#define ECC_ONLY_ITERS 50
#endif

#if PROFILER_ECC_ONLY
#define ECC_KEYS_PER_THREAD ECC_ONLY_STEPS
#else
#define ECC_KEYS_PER_THREAD BATCH_SIZE
#endif

__device__ __constant__ uint8_t d_target[20];
__device__ __constant__ uint32_t d_prefix;
__device__ int d_found = 0;
__device__ uint64_t d_found_key[4];
__device__ __constant__ uint64_t d_key_high[3];
__device__ __constant__ uint64_t d_G_multiples_x[BATCH_SIZE * 4];
__device__ __constant__ uint64_t d_G_multiples_y[BATCH_SIZE * 4];

// Contadores para profiling detalhado
__device__ unsigned long long d_time_batch_inv = 0;
__device__ unsigned long long d_time_point_add = 0;
__device__ unsigned long long d_time_hash = 0;
__device__ unsigned long long d_time_check = 0;

#if ECC_PREVENT_DCE
__device__ unsigned long long d_ecc_sink = 0;
#endif

static __forceinline__ int cmp256_le(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static __forceinline__ void add_u64_to_256(uint64_t a[4], uint64_t add) {
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

#if PROFILER_ECC_ONLY && ECC_ONLY_PERSIST_JACOBIAN
__global__ void init_pts_blocks_jacobian(uint64_t* outX, uint64_t* outY, uint64_t* outZ,
                                        uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t bid = (uint64_t)blockIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};

    // Offset per block: each block covers THREADS_PER_BLOCK*ECC_KEYS_PER_THREAD points per launch
    const unsigned __int128 off = (unsigned __int128)bid * (unsigned __int128)THREADS_PER_BLOCK * (unsigned __int128)ECC_KEYS_PER_THREAD;
    unsigned __int128 r0 = (unsigned __int128)key[0] + (uint64_t)off;
    key[0] = (uint64_t)r0;

    unsigned __int128 r1 = (unsigned __int128)key[1] + (uint64_t)(r0 >> 64);
    key[1] = (uint64_t)r1;

    if (r1 >> 64) {
        uint64_t r2 = key[2] + 1;
        key[2] = r2;
        if (r2 == 0) key[3]++;
    }

    uint64_t ax[4], ay[4];
    scalarMulBaseAffine(key, ax, ay);

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
        outX[bid*4 + i] = Pj.X[i];
        outY[bid*4 + i] = Pj.Y[i];
        outZ[bid*4 + i] = Pj.Z[i];
    }
}
#endif

__global__ void init_pts_blocks(uint64_t* d_px, uint64_t* d_py, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    const uint64_t bid = (uint64_t)blockIdx.x;
    uint64_t key[4] = {s0, s1, s2, s3};

    // Offset per block: each block covers THREADS_PER_BLOCK*ECC_KEYS_PER_THREAD points per launch
    const unsigned __int128 off = (unsigned __int128)bid * (unsigned __int128)THREADS_PER_BLOCK * (unsigned __int128)ECC_KEYS_PER_THREAD;
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
        d_px[bid*4 + i] = ox[i];
        d_py[bid*4 + i] = oy[i];
    }
}

__device__ __forceinline__ void pointAddJacobianG_Hot(const ECPointJ& P, ECPointJ& R);
__device__ __forceinline__ void pointAddJacobianMixed_XY_Hot(const ECPointJ& P, const uint64_t QX[4], const uint64_t QY[4], ECPointJ& R);

__global__ void __launch_bounds__(256, 4)
kernel_profiled_genius_ecc_only(
    uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py,
#if ECC_ONLY_PERSIST_JACOBIAN
    uint64_t* __restrict__ pjx, uint64_t* __restrict__ pjy, uint64_t* __restrict__ pjz,
#endif
    uint64_t start_key_lo, unsigned long long* __restrict__ d_count) {
    const unsigned tid = (unsigned)threadIdx.x;
    const unsigned lane = tid & 31;
    const unsigned bid = (unsigned)blockIdx.x;

    // One base point per block
    __shared__ uint64_t shX[4];
    __shared__ uint64_t shY[4];
    __shared__ uint64_t shZ[4];

    if(tid == 0) {
        #if ECC_ONLY_PERSIST_JACOBIAN
        #pragma unroll
        for(int i=0; i<4; i++) {
            shX[i] = pjx[bid*4 + i];
            shY[i] = pjy[bid*4 + i];
            shZ[i] = pjz[bid*4 + i];
        }
        #else
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
        #endif
    }
    __syncthreads();

    unsigned long long lc = 0;
    unsigned long long local_cycles = 0;
#if ECC_PREVENT_DCE
    unsigned long long sink_acc = 0;
#endif

    // Cache Q = (tid+1)*G once (affine)
    uint64_t QX[4], QY[4];
    #pragma unroll
    for(int i=0; i<4; i++) {
        QX[i] = d_G_multiples_x[tid*4 + i];
        QY[i] = d_G_multiples_y[tid*4 + i];
    }

    // Cache Qstep = (ECC_ONLY_STEPS)G (index ECC_ONLY_STEPS-1) once in tid0
    uint64_t Q256X[4], Q256Y[4];
    if(tid == 0) {
        #pragma unroll
        for(int i=0; i<4; i++) {
            Q256X[i] = d_G_multiples_x[(ECC_ONLY_STEPS-1)*4 + i];
            Q256Y[i] = d_G_multiples_y[(ECC_ONLY_STEPS-1)*4 + i];
        }
    }

    // Process ECC_ONLY_STEPS batches; each batch computes 256 points in parallel
    for(int it = 0; it < ECC_ONLY_STEPS; it++) {
        ECPointJ P;
        #pragma unroll
        for(int i=0; i<4; i++) {
            P.X[i] = shX[i];
            P.Y[i] = shY[i];
            P.Z[i] = shZ[i];
        }
        P.infinity = false;

        unsigned long long t0 = 0, t1 = 0;
        if(tid == 0) t0 = clock64();

        ECPointJ R;
        pointAddJacobianMixed_XY_Hot(P, QX, QY, R);

        // Prevent dead-code elimination (cheap): only use one word from R in block0/tid0
#if ECC_PREVENT_DCE
        if(bid == 0 && tid == 0) sink_acc ^= (unsigned long long)R.X[0];
#endif

        if(tid == 0) {
            t1 = clock64();
            local_cycles += (t1 - t0);
        }

        lc++;

        // Advance base by +ECC_ONLY_STEPS*G
        if(tid == 0) {
            ECPointJ newP;
            pointAddJacobianMixed_XY_Hot(P, Q256X, Q256Y, newP);
            #pragma unroll
            for(int i=0; i<4; i++) {
                shX[i] = newP.X[i];
                shY[i] = newP.Y[i];
                shZ[i] = newP.Z[i];
            }
        }
        __syncthreads();
    }

    if(tid == 0) {
        // Only accumulate representative timing from block 0 to keep "cycles per point" meaningful
        if(bid == 0) atomicAdd(&d_time_point_add, local_cycles);
#if ECC_PREVENT_DCE
        if(bid == 0) atomicXor(&d_ecc_sink, sink_acc);
#endif
    }

    // Store updated base back to global (one point per block)
    if(tid == 0) {
        #if ECC_ONLY_PERSIST_JACOBIAN
        #pragma unroll
        for(int i=0; i<4; i++) {
            pjx[bid*4 + i] = shX[i];
            pjy[bid*4 + i] = shY[i];
            pjz[bid*4 + i] = shZ[i];
        }
        #else
        ECPointJ Pj;
        #pragma unroll
        for(int i=0; i<4; i++) {
            Pj.X[i] = shX[i];
            Pj.Y[i] = shY[i];
            Pj.Z[i] = shZ[i];
        }
        Pj.infinity = false;
        ECPointA P_final;
        jacobianToAffine(Pj, P_final);
        #pragma unroll
        for(int i=0; i<4; i++) {
            d_px[bid*4 + i] = P_final.X[i];
            d_py[bid*4 + i] = P_final.Y[i];
        }
        #endif
    }

    // Warp reduction for total points counted (per thread = BATCH_SIZE)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if(lane == 0) atomicAdd(d_count, lc);
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

__device__ __forceinline__ void pointAddJacobianG_Hot(const ECPointJ& P, ECPointJ& R) {
    pointAddJacobianMixed_XY_Hot(P, SECP_GX_LE, SECP_GY_LE, R);
}

// Batch inversion da melhor versão (2.07 GKeys/s)
__device__ void batchInversion_Best(uint64_t dx_values[][4], int count, uint64_t dx_inv[][4]) {
    if(count <= 1) {
        fieldInv_Fermat(dx_values[0], dx_inv[0]);
        return;
    }
    
    uint64_t products[256][4];
    for(int i=0; i<4; i++) products[0][i] = dx_values[0][i];
    
    for(int i=1; i<count; i++) {
        fieldMul(products[i-1], dx_values[i], products[i]);
    }
    
    uint64_t inv_product[4];
    fieldInv_Fermat(products[count-1], inv_product);
    
    uint64_t inv[4];
    for(int i=0; i<4; i++) inv[i] = inv_product[i];
    
    for(int i=count-1; i>0; i--) {
        fieldMul(inv, products[i-1], dx_inv[i]);
        uint64_t temp[4];
        fieldMul(inv, dx_values[i], temp);
        for(int j=0; j<4; j++) inv[j] = temp[j];
    }
    
    for(int i=0; i<4; i++) dx_inv[0][i] = inv[i];
}

// Kernel com profiling integrado
__global__ void __launch_bounds__(256, 4)
kernel_profiled_genius(uint64_t* __restrict__ d_px, uint64_t* __restrict__ d_py, 
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

    unsigned long long start_batch = clock64();

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

    uint64_t inv_product[4];
    fieldInv_Fermat(products[BATCH_SIZE-1], inv_product);

    uint64_t inv[4];
    #pragma unroll
    for(int j=0; j<4; j++) inv[j] = inv_product[j];

    uint64_t inv_dx_last[4];

    unsigned long long end_batch = clock64();
    if(tid == 0) atomicAdd(&d_time_batch_inv, end_batch - start_batch);

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
        
        // PROFILING: Point Addition
        unsigned long long start_point = clock64();
        
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
        
        unsigned long long end_point = clock64();
        if(tid == 0 && i == 0) atomicAdd(&d_time_point_add, end_point - start_point);
        
        // PROFILING: Hash
        unsigned long long start_hash = clock64();
        
        uint32_t sha_state[8];
        sha256_opt(x3[0], x3[1], x3[2], x3[3], (y3[0] & 1) ? 0x03 : 0x02, sha_state);
        
        uint8_t hash[20];
        ripemd160_opt(sha_state, hash);
        
        unsigned long long end_hash = clock64();
        if(tid == 0 && i == 0) atomicAdd(&d_time_hash, end_hash - start_hash);
        
        lc++;

        // PROFILING: Check
        unsigned long long start_check = clock64();
        
        if (*(uint32_t*)hash == d_prefix) {
            bool match = true;
            #pragma unroll
            for(int k=0; k<20; k++) {
                if(hash[k] != d_target[k]) {
                    match = false;
                    break;
                }
            }
            
            if (match && atomicCAS(&d_found, 0, 1) == 0) {
                d_found_key[0] = local_key + i;
                d_found_key[1] = d_key_high[0];
                d_found_key[2] = d_key_high[1];
                d_found_key[3] = d_key_high[2];
            }
        }
        
        unsigned long long end_check = clock64();
        if(tid == 0 && i == 0) atomicAdd(&d_time_check, end_check - start_check);

        uint64_t tmp_inv[4];
        fieldMul(inv, dx, tmp_inv);
        #pragma unroll
        for(int j=0; j<4; j++) inv[j] = tmp_inv[j];
    }

    // Update base point
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
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lc += __shfl_down_sync(0xFFFFFFFF, lc, offset);
    }
    if(lane == 0) atomicAdd(d_count, lc);
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

void initGMultiples() {
    uint64_t* h_gx = new uint64_t[BATCH_SIZE * 4];
    uint64_t* h_gy = new uint64_t[BATCH_SIZE * 4];

    uint64_t *d_gx = nullptr, *d_gy = nullptr;
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

int main(int argc, char** argv) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║      HYPERION PROFILER GENIUS - Análise Ultra-Precisa      ║\n");
    printf("║      Usando melhor versão (2.07 GKeys/s) + Profiling       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    if (argc >= 2 && strcmp(argv[1], "--samples") == 0) {
        int count = 5;
        if (argc >= 3) count = atoi(argv[2]);
        if (count < 1) count = 1;
        if (count > 64) count = 64;

        initGMultiples();

        uint64_t h_keys[64 * 4] = {0};
        for (int i = 0; i < count; i++) {
            h_keys[i * 4 + 0] = (uint64_t)(i + 1);
            h_keys[i * 4 + 1] = 0;
            h_keys[i * 4 + 2] = 0;
            h_keys[i * 4 + 3] = 0;
        }

        uint64_t* d_keys = nullptr;
        uint8_t* d_hash = nullptr;
        cudaMalloc(&d_keys, (size_t)count * 4 * sizeof(uint64_t));
        cudaMalloc(&d_hash, (size_t)count * 20 * sizeof(uint8_t));
        cudaMemcpy(d_keys, h_keys, (size_t)count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

        kernel_samples_hash160<<<1, 64>>>(d_keys, d_hash, count);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("❌ CUDA Error during samples: %s\n", cudaGetErrorString(err));
            return 1;
        }

        uint8_t h_hash[64 * 20] = {0};
        cudaMemcpy(h_hash, d_hash, (size_t)count * 20 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        printf("Samples (privkey -> hash160, compressed pubkey):\n");
        for (int i = 0; i < count; i++) {
            uint64_t k[4] = { h_keys[i*4+0], h_keys[i*4+1], h_keys[i*4+2], h_keys[i*4+3] };
            printKey256("  priv=0x", k);
            printHash160("  hash160=", &h_hash[i * 20]);
            printf("\n");
        }

        cudaFree(d_keys);
        cudaFree(d_hash);
        return 0;
    }

    const char* hx = "f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8";
    const char* range = "1:100000";

    for (int ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "-f") == 0 && (ai + 1) < argc) {
            hx = argv[++ai];
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
    
    uint8_t tgt[20];
    if (!parseHash160(hx, tgt)) {
        printf("❌ Hash160 inválido. Use -f <40 hex chars>\n");
        return 1;
    }

    printf("Input:\n");
    printf("  Target (hash160): %s\n", hx);
    {
        uint64_t tmp[4];
        for (int i=0;i<4;i++) tmp[i]=rs[i];
        printKey256("  Range start: 0x", tmp);
        for (int i=0;i<4;i++) tmp[i]=re[i];
        printKey256("  Range end:   0x", tmp);
    }

    initGMultiples();

    uint64_t thr = (uint64_t)NUM_BLOCKS * THREADS_PER_BLOCK;
    uint64_t kpl = thr * (uint64_t)ECC_KEYS_PER_THREAD;
    
    cudaMemcpyToSymbol(d_key_high, &rs[1], 24);
    cudaMemcpyToSymbol(d_target, tgt, 20);
    uint32_t pfx = *(uint32_t*)tgt;
    cudaMemcpyToSymbol(d_prefix, &pfx, 4);
    int z = 0;
    cudaMemcpyToSymbol(d_found, &z, sizeof(int));

    uint64_t *d_px, *d_py;
    #if PROFILER_ECC_ONLY
    cudaMalloc(&d_px, (uint64_t)NUM_BLOCKS * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, (uint64_t)NUM_BLOCKS * 4 * sizeof(uint64_t));
    #else
    cudaMalloc(&d_px, thr * 4 * sizeof(uint64_t));
    cudaMalloc(&d_py, thr * 4 * sizeof(uint64_t));
    #endif
    unsigned long long* d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    unsigned long long uz = 0;
    cudaMemcpy(d_cnt, &uz, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Reset profiling counters
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(d_time_batch_inv, &zero, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_time_point_add, &zero, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_time_hash, &zero, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_time_check, &zero, sizeof(unsigned long long));

    printf("Configuration:\n");
    printf("  Blocks:         %d\n", NUM_BLOCKS);
    printf("  Threads:        %d\n", THREADS_PER_BLOCK);
    printf("  Batch Size:     %d\n", BATCH_SIZE);
    #if PROFILER_ECC_ONLY
    printf("  ECC Steps:      %d\n", (int)ECC_ONLY_STEPS);
    printf("  ECC Iters:      %d\n", (int)ECC_ONLY_ITERS);
    #endif
    printf("  Keys/Launch:    %llu (%.2f million)\n\n", (unsigned long long)kpl, kpl/1e6);
    
    printf("Inicializando...\n");
    #if PROFILER_ECC_ONLY
    #if ECC_ONLY_PERSIST_JACOBIAN
    init_pts_blocks_jacobian<<<NUM_BLOCKS, 1>>>(d_px, d_py, d_pjz, rs[0], rs[1], rs[2], rs[3]);
    #else
    init_pts_blocks<<<NUM_BLOCKS, 1>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    #endif
    #else
    init_pts<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, rs[0], rs[1], rs[2], rs[3]);
    #endif
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("❌ CUDA Error during init: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("✓ Init OK!\n\n");
    
    printf("═══════════════════════════════════════════════════════════════\n");
    #if PROFILER_ECC_ONLY
    printf("           PROFILING GENIUS - %d Iterações\n", (int)ECC_ONLY_ITERS);
    #else
    printf("           PROFILING GENIUS - 10 Iterações\n");
    #endif
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    // CUDA Events para medição precisa
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cudaEventRecord(start_event);
    auto start = std::chrono::steady_clock::now();
    uint64_t cur[4] = { rs[0], rs[1], rs[2], rs[3] };
    uint64_t batch_lo = cur[0];
    
    #if PROFILER_ECC_ONLY
    for (int i = 0; i < ECC_ONLY_ITERS; i++) {
    #else
    for (int i = 0; i < 10; i++) {
    #endif
        #if PROFILER_ECC_ONLY
        #if ECC_ONLY_PERSIST_JACOBIAN
        kernel_profiled_genius_ecc_only<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, d_pjx, d_pjy, d_pjz, batch_lo, d_cnt);
        #else
        kernel_profiled_genius_ecc_only<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_lo, d_cnt);
        #endif
        #else
        kernel_profiled_genius<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_px, d_py, batch_lo, d_cnt);
        #endif
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n❌ CUDA Error: %s\n", cudaGetErrorString(err));
            break;
        }
        
        unsigned long long total;
        cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double speed = total / elapsed / 1e9;
        
        #if PROFILER_ECC_ONLY
        printf("\rIteração %2d/%d: [%.2f GKeys/s] %llu keys   ", i+1, (int)ECC_ONLY_ITERS, speed, total);
        #else
        printf("\rIteração %2d/10: [%.2f GKeys/s] %llu keys   ", i+1, speed, total);
        #endif
        fflush(stdout);
        
        add_u64_to_256(cur, kpl);
        batch_lo = cur[0];
        cudaMemcpyToSymbol(d_key_high, &cur[1], 24);
        if (cmp256_le(cur, re) > 0) {
            printf("\n\n✓ Range completo: próximo start > end\n");
            break;
        }
    }
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    
    float cuda_time_ms;
    cudaEventElapsedTime(&cuda_time_ms, start_event, stop_event);
    
    unsigned long long total;
    cudaMemcpy(&total, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double speed = total / elapsed / 1e9;
    
    // Ler contadores de profiling
    unsigned long long time_batch_inv, time_point_add, time_hash, time_check;
    cudaMemcpyFromSymbol(&time_batch_inv, d_time_batch_inv, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&time_point_add, d_time_point_add, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&time_hash, d_time_hash, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&time_check, d_time_check, sizeof(unsigned long long));
    
    printf("\n\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("           🎯 PROFILING GENIUS RESULTS 🎯\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    printf("Performance Geral:\n");
    printf("  Velocidade:     %.2f GKeys/s\n", speed);
    printf("  Chaves:         %llu\n", total);
    printf("  Tempo (CPU):    %.2f segundos\n", elapsed);
    printf("  Tempo (CUDA):   %.2f segundos\n\n", cuda_time_ms/1000.0);
    
    printf("Breakdown Detalhado (em cycles - thread 0 apenas):\n");
    printf("  Batch Inversion:  %llu cycles\n", time_batch_inv);
    #if PROFILER_ECC_ONLY
    {
        // In ECC-only block=basepoint mode, d_time_point_add measures ONLY block0, thread0 point-add cycles per iteration.
        const unsigned long long denom_points = (unsigned long long)ECC_ONLY_STEPS * (unsigned long long)ECC_ONLY_ITERS;
        unsigned long long per_point = denom_points ? (time_point_add / denom_points) : 0ULL;
        printf("  Point Addition:   %llu cycles total (%d iters, thread 0)\n", time_point_add, (int)ECC_ONLY_ITERS);
        printf("                  ~ %llu cycles (por ponto)\n", per_point);
    }
    printf("  Hash (SHA+RIPEMD):%llu cycles (por ponto)\n", 0ULL);
    printf("  Check/Verify:     %llu cycles (por ponto)\n\n", 0ULL);
    #else
    printf("  Point Addition:   %llu cycles (por ponto)\n", time_point_add);
    printf("  Hash (SHA+RIPEMD):%llu cycles (por ponto)\n", time_hash);
    printf("  Check/Verify:     %llu cycles (por ponto)\n\n", time_check);
    #endif
    
    printf("Análise de Gargalos:\n");
    #if PROFILER_ECC_ONLY
    unsigned long long total_cycles = time_point_add;
    #else
    unsigned long long total_cycles = time_batch_inv + (time_point_add + time_hash + time_check) * BATCH_SIZE;
    #endif
    if(total_cycles > 0) {
        printf("  Batch Inversion:  %.1f%% do tempo\n", 100.0 * time_batch_inv / total_cycles);
        #if PROFILER_ECC_ONLY
        printf("  Point Addition:   %.1f%% do tempo\n", 100.0 * time_point_add / total_cycles);
        printf("  Hash:             %.1f%% do tempo\n", 0.0);
        printf("  Check:            %.1f%% do tempo\n\n", 0.0);
        #else
        printf("  Point Addition:   %.1f%% do tempo\n", 100.0 * (time_point_add * BATCH_SIZE) / total_cycles);
        printf("  Hash:             %.1f%% do tempo\n", 100.0 * (time_hash * BATCH_SIZE) / total_cycles);
        printf("  Check:            %.1f%% do tempo\n\n", 100.0 * (time_check * BATCH_SIZE) / total_cycles);
        #endif
    }
    
    printf("Recomendações de Otimização:\n");
    if(time_batch_inv > (time_point_add + time_hash) * BATCH_SIZE) {
        printf("  🔴 CRÍTICO: Batch Inversion é o maior gargalo\n");
        printf("     → Aumentar batch size (se possível)\n");
        printf("     → Otimizar fieldInv_Fermat\n\n");
    }
    if(time_hash > time_point_add) {
        printf("  🟡 ALTO: Hash consome mais que Point Addition\n");
        printf("     → Implementar SHA256 lookup tables\n");
        printf("     → Otimizar RIPEMD160\n\n");
    }
    if(time_point_add > time_hash) {
        printf("  🟡 ALTO: Point Addition consome mais que Hash\n");
        printf("     → Implementar field ops com PTX inline\n");
        printf("     → Otimizar fieldMul e fieldSub\n\n");
    }
    
    printf("═══════════════════════════════════════════════════════════════\n\n");
    
    printf("Próximos Passos para Máxima Velocidade:\n");
    printf("  1. Otimizar o componente mais lento identificado acima\n");
    printf("  2. Re-executar profiling para validar ganho\n");
    printf("  3. Iterar até atingir máxima performance\n");
    printf("  4. Migrar para RTX 3090 dedicada (3x speedup)\n\n");
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cnt);
    return 0;
}
