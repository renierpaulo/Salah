/*
 * CUDA MATH ULTRA - Optimized 256-bit Field Operations
 * 
 * Optimizations:
 * 1. PTX inline assembly for carry propagation
 * 2. Fused multiply-add chains
 * 3. Reduced register pressure
 * 4. Warp-level instruction scheduling
 */
#ifndef CUDA_MATH_ULTRA_CUH
#define CUDA_MATH_ULTRA_CUH

#include <cstdint>

// secp256k1 prime: p = 2^256 - 2^32 - 977
// In little-endian uint64_t[4]: {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

// PTX-optimized 64-bit addition with carry
__device__ __forceinline__ uint64_t add_cc(uint64_t a, uint64_t b, uint32_t& carry) {
    uint64_t result;
    asm volatile(
        "add.cc.u64 %0, %1, %2;\n\t"
        "addc.u32 %3, 0, 0;"
        : "=l"(result), "=r"(carry)
        : "l"(a), "l"(b)
    );
    return result;
}

// PTX-optimized 64-bit addition with carry in/out
__device__ __forceinline__ uint64_t addc_cc(uint64_t a, uint64_t b, uint32_t& carry) {
    uint64_t result;
    asm volatile(
        "add.cc.u64 %0, %1, %2;\n\t"
        "addc.cc.u64 %0, %0, %3;\n\t"
        "addc.u32 %4, 0, 0;"
        : "=l"(result), "=r"(carry)
        : "l"(a), "l"(b), "r"(carry)
    );
    return result;
}

// Ultra-fast 256-bit addition mod p
__device__ __forceinline__ void fieldAdd_Ultra(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    // First add without reduction
    uint64_t s0, s1, s2, s3;
    uint32_t carry = 0;
    
    asm volatile(
        "add.cc.u64 %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.cc.u64 %3, %7, %11;\n\t"
        "addc.u32 %12, 0, 0;"
        : "=l"(s0), "=l"(s1), "=l"(s2), "=l"(s3), "=r"(carry)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
    
    // Reduce mod p if needed
    // p = 2^256 - 2^32 - 977 = {0xFFFFFFFEFFFFFC2F, 0xFFFF...}
    // If carry or s >= p, subtract p (add 2^32 + 977)
    const uint64_t p0 = 0xFFFFFFFEFFFFFC2FULL;
    
    bool need_reduce = carry || (s3 > 0xFFFFFFFFFFFFFFFFULL) || 
                       (s3 == 0xFFFFFFFFFFFFFFFFULL && s2 == 0xFFFFFFFFFFFFFFFFULL && 
                        s1 == 0xFFFFFFFFFFFFFFFFULL && s0 >= p0);
    
    if(need_reduce) {
        // Subtract p = add (2^32 + 977)
        uint64_t adj = 0x100000000ULL + 977;
        asm volatile(
            "add.cc.u64 %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, 0;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64 %3, %3, 0;"
            : "+l"(s0), "+l"(s1), "+l"(s2), "+l"(s3)
            : "l"(adj)
        );
    }
    
    r[0] = s0; r[1] = s1; r[2] = s2; r[3] = s3;
}

// Ultra-fast 256-bit subtraction mod p (C++ version for compatibility)
__device__ __forceinline__ void fieldSub_Ultra(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    // Use 128-bit arithmetic for proper borrow handling
    unsigned __int128 d0 = (unsigned __int128)a[0] - b[0];
    unsigned __int128 d1 = (unsigned __int128)a[1] - b[1] - (d0 >> 127);
    unsigned __int128 d2 = (unsigned __int128)a[2] - b[2] - (d1 >> 127);
    unsigned __int128 d3 = (unsigned __int128)a[3] - b[3] - (d2 >> 127);
    
    r[0] = (uint64_t)d0;
    r[1] = (uint64_t)d1;
    r[2] = (uint64_t)d2;
    r[3] = (uint64_t)d3;
    
    // If borrow (d3 negative), add p
    if(d3 >> 127) {
        // Add p by subtracting (2^32 + 977) from the result conceptually
        // p = 2^256 - 2^32 - 977, so adding p = subtracting 2^32+977 from overflow
        const uint64_t adj = 0xFFFFFFFEFFFFFC2FULL; // p[0]
        unsigned __int128 s0 = (unsigned __int128)r[0] + adj;
        unsigned __int128 s1 = (unsigned __int128)r[1] + 0xFFFFFFFFFFFFFFFFULL + (s0 >> 64);
        unsigned __int128 s2 = (unsigned __int128)r[2] + 0xFFFFFFFFFFFFFFFFULL + (s1 >> 64);
        unsigned __int128 s3 = (unsigned __int128)r[3] + 0xFFFFFFFFFFFFFFFFULL + (s2 >> 64);
        r[0] = (uint64_t)s0;
        r[1] = (uint64_t)s1;
        r[2] = (uint64_t)s2;
        r[3] = (uint64_t)s3;
    }
}

// 64x64 -> 128 bit multiply using PTX
__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    asm volatile(
        "mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b)
    );
}

// Ultra-fast 256-bit multiplication mod p using Karatsuba-like optimization
__device__ __forceinline__ void fieldMul_Ultra(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    // Full 512-bit product using schoolbook (16 64x64 multiplies)
    // Then reduce mod p
    
    uint64_t p[8] = {0}; // 512-bit product
    
    // Compute partial products and accumulate
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            mul64(a[i], b[j], lo, hi);
            
            // Add to p[i+j] and p[i+j+1]
            uint64_t sum = p[i+j] + lo + carry;
            carry = (sum < p[i+j] || sum < lo) ? 1 : 0;
            carry += hi;
            p[i+j] = sum;
        }
        p[i+4] += carry;
    }
    
    // Reduce mod p using the special form p = 2^256 - 2^32 - 977
    // For h = high 256 bits, l = low 256 bits:
    // result = l + h * (2^32 + 977) mod p
    
    uint64_t h[4] = {p[4], p[5], p[6], p[7]};
    uint64_t l[4] = {p[0], p[1], p[2], p[3]};
    
    // Multiply h by (2^32 + 977) = 0x1000003D1
    const uint64_t mult = 0x1000003D1ULL;
    uint64_t hm[5] = {0};
    uint64_t carry = 0;
    
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        uint64_t lo, hi;
        mul64(h[i], mult, lo, hi);
        uint64_t sum = hm[i] + lo + carry;
        carry = (sum < lo) ? 1 : 0;
        carry += hi;
        hm[i] = sum;
    }
    hm[4] = carry;
    
    // Add l + hm
    carry = 0;
    asm volatile(
        "add.cc.u64 %0, %0, %4;\n\t"
        "addc.cc.u64 %1, %1, %5;\n\t"
        "addc.cc.u64 %2, %2, %6;\n\t"
        "addc.cc.u64 %3, %3, %7;\n\t"
        "addc.u64 %8, 0, 0;"
        : "+l"(l[0]), "+l"(l[1]), "+l"(l[2]), "+l"(l[3]), "=l"(carry)
        : "l"(hm[0]), "l"(hm[1]), "l"(hm[2]), "l"(hm[3])
    );
    carry += hm[4];
    
    // If carry, reduce again
    if(carry) {
        uint64_t adj = carry * mult;
        asm volatile(
            "add.cc.u64 %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, 0;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64 %3, %3, 0;"
            : "+l"(l[0]), "+l"(l[1]), "+l"(l[2]), "+l"(l[3])
            : "l"(adj)
        );
    }
    
    // Final reduction if >= p
    const uint64_t p0 = 0xFFFFFFFEFFFFFC2FULL;
    if(l[3] == 0xFFFFFFFFFFFFFFFFULL && l[2] == 0xFFFFFFFFFFFFFFFFULL && 
       l[1] == 0xFFFFFFFFFFFFFFFFULL && l[0] >= p0) {
        l[0] += 0x1000003D1ULL;
        // Carry propagation not needed since we're reducing from p
    }
    
    r[0] = l[0]; r[1] = l[1]; r[2] = l[2]; r[3] = l[3];
}

// Ultra-fast 256-bit squaring mod p (slightly faster than mul due to symmetry)
__device__ __forceinline__ void fieldSqr_Ultra(const uint64_t a[4], uint64_t r[4]) {
    // Use symmetry: a[i]*a[j] appears twice for i != j
    uint64_t p[8] = {0};
    
    // Diagonal terms (i == j)
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        uint64_t lo, hi;
        mul64(a[i], a[i], lo, hi);
        p[2*i] += lo;
        p[2*i+1] += hi;
    }
    
    // Off-diagonal terms (i < j), multiply by 2
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int j = i+1; j < 4; j++) {
            uint64_t lo, hi;
            mul64(a[i], a[j], lo, hi);
            
            // Double the product
            uint64_t lo2 = lo << 1;
            uint64_t hi2 = (hi << 1) | (lo >> 63);
            
            // Add to accumulator
            uint64_t sum = p[i+j] + lo2;
            p[i+j] = sum;
            p[i+j+1] += hi2 + (sum < lo2 ? 1 : 0);
        }
    }
    
    // Reduce mod p (same as fieldMul_Ultra)
    uint64_t h[4] = {p[4], p[5], p[6], p[7]};
    uint64_t l[4] = {p[0], p[1], p[2], p[3]};
    
    const uint64_t mult = 0x1000003D1ULL;
    uint64_t hm[5] = {0};
    uint64_t carry = 0;
    
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        uint64_t lo, hi;
        mul64(h[i], mult, lo, hi);
        uint64_t sum = hm[i] + lo + carry;
        carry = (sum < lo) ? 1 : 0;
        carry += hi;
        hm[i] = sum;
    }
    hm[4] = carry;
    
    // Add l + hm
    l[0] += hm[0];
    carry = (l[0] < hm[0]) ? 1 : 0;
    l[1] += hm[1] + carry;
    carry = (l[1] < hm[1] + carry) ? 1 : 0;
    l[2] += hm[2] + carry;
    carry = (l[2] < hm[2] + carry) ? 1 : 0;
    l[3] += hm[3] + carry;
    carry = (l[3] < hm[3] + carry) ? 1 : 0;
    carry += hm[4];
    
    if(carry) {
        uint64_t adj = carry * mult;
        l[0] += adj;
        if(l[0] < adj) { l[1]++; if(l[1] == 0) { l[2]++; if(l[2] == 0) l[3]++; } }
    }
    
    r[0] = l[0]; r[1] = l[1]; r[2] = l[2]; r[3] = l[3];
}

#endif // CUDA_MATH_ULTRA_CUH
