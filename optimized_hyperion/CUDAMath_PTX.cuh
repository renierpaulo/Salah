/*
 * CUDAMath_PTX.cuh - Ultra-optimized field operations using PTX inline assembly
 * Target: secp256k1 field (p = 2^256 - 2^32 - 977)
 * Goal: 9x speedup for Point Addition
 */

#pragma once
#include <cstdint>

// secp256k1 prime: p = 2^256 - 2^32 - 977
// In 64-bit limbs: {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

// Ultra-fast field subtraction using PTX
__device__ __forceinline__ void fieldSub_PTX(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t borrow;
    asm volatile (
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.u64 %3, %7, %11;\n\t"
        : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
    
    // Check if borrow occurred (result negative) and add p back
    int64_t sign = (int64_t)r[3] >> 63;
    uint64_t mask = (uint64_t)sign;
    
    // Add p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
    asm volatile (
        "add.cc.u64 %0, %0, %4;\n\t"
        "addc.cc.u64 %1, %1, %5;\n\t"
        "addc.cc.u64 %2, %2, %6;\n\t"
        "addc.u64 %3, %3, %7;\n\t"
        : "+l"(r[0]), "+l"(r[1]), "+l"(r[2]), "+l"(r[3])
        : "l"(mask & 0xFFFFFFFEFFFFFC2FULL), "l"(mask), "l"(mask), "l"(mask)
    );
}

// Ultra-fast field addition using PTX
__device__ __forceinline__ void fieldAdd_PTX(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry;
    asm volatile (
        "add.cc.u64 %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.u64 %3, %7, %11;\n\t"
        : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
    
    // Reduce if >= p
    uint64_t t[4];
    asm volatile (
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.u64 %3, %7, %11;\n\t"
        : "=l"(t[0]), "=l"(t[1]), "=l"(t[2]), "=l"(t[3])
        : "l"(r[0]), "l"(r[1]), "l"(r[2]), "l"(r[3]),
          "l"(0xFFFFFFFEFFFFFC2FULL), "l"(0xFFFFFFFFFFFFFFFFULL), 
          "l"(0xFFFFFFFFFFFFFFFFULL), "l"(0xFFFFFFFFFFFFFFFFULL)
    );
    
    // Select r or t based on whether subtraction underflowed
    int64_t sign = (int64_t)t[3] >> 63;
    uint64_t mask = (uint64_t)sign;
    r[0] = (r[0] & mask) | (t[0] & ~mask);
    r[1] = (r[1] & mask) | (t[1] & ~mask);
    r[2] = (r[2] & mask) | (t[2] & ~mask);
    r[3] = (r[3] & mask) | (t[3] & ~mask);
}

// 256x256 -> 512 bit multiplication using PTX mad.hi/mad.lo
__device__ __forceinline__ void mul256x256_PTX(const uint64_t a[4], const uint64_t b[4], uint64_t r[8]) {
    uint64_t t0, t1, t2, t3, t4, t5, t6, t7;
    uint64_t c0, c1, c2;
    
    // Column 0
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(t0) : "l"(a[0]), "l"(b[0]));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(c0) : "l"(a[0]), "l"(b[0]));
    
    // Column 1
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(t1) : "l"(a[0]), "l"(b[1]));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(c1) : "l"(a[0]), "l"(b[1]));
    asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(t1) : "l"(c0));
    asm volatile ("addc.u64 %0, %0, 0;" : "+l"(c1));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t1) : "l"(a[1]), "l"(b[0]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c1) : "l"(a[1]), "l"(b[0]));
    
    // Column 2
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(t2) : "l"(a[0]), "l"(b[2]));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(c2) : "l"(a[0]), "l"(b[2]));
    asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(t2) : "l"(c1));
    asm volatile ("addc.u64 %0, %0, 0;" : "+l"(c2));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t2) : "l"(a[1]), "l"(b[1]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c2) : "l"(a[1]), "l"(b[1]));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t2) : "l"(a[2]), "l"(b[0]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c2) : "l"(a[2]), "l"(b[0]));
    
    // Column 3
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(t3) : "l"(a[0]), "l"(b[3]));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(c0) : "l"(a[0]), "l"(b[3]));
    asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(t3) : "l"(c2));
    asm volatile ("addc.u64 %0, %0, 0;" : "+l"(c0));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t3) : "l"(a[1]), "l"(b[2]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c0) : "l"(a[1]), "l"(b[2]));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t3) : "l"(a[2]), "l"(b[1]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c0) : "l"(a[2]), "l"(b[1]));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t3) : "l"(a[3]), "l"(b[0]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c0) : "l"(a[3]), "l"(b[0]));
    
    // Column 4
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(t4) : "l"(a[1]), "l"(b[3]));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(c1) : "l"(a[1]), "l"(b[3]));
    asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(t4) : "l"(c0));
    asm volatile ("addc.u64 %0, %0, 0;" : "+l"(c1));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t4) : "l"(a[2]), "l"(b[2]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c1) : "l"(a[2]), "l"(b[2]));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t4) : "l"(a[3]), "l"(b[1]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c1) : "l"(a[3]), "l"(b[1]));
    
    // Column 5
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(t5) : "l"(a[2]), "l"(b[3]));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(c2) : "l"(a[2]), "l"(b[3]));
    asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(t5) : "l"(c1));
    asm volatile ("addc.u64 %0, %0, 0;" : "+l"(c2));
    
    asm volatile ("mad.lo.cc.u64 %0, %1, %2, %0;" : "+l"(t5) : "l"(a[3]), "l"(b[2]));
    asm volatile ("madc.hi.u64 %0, %1, %2, %0;" : "+l"(c2) : "l"(a[3]), "l"(b[2]));
    
    // Column 6
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(t6) : "l"(a[3]), "l"(b[3]));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(t7) : "l"(a[3]), "l"(b[3]));
    asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(t6) : "l"(c2));
    asm volatile ("addc.u64 %0, %0, 0;" : "+l"(t7));
    
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
    r[4] = t4; r[5] = t5; r[6] = t6; r[7] = t7;
}

// Fast modular reduction for secp256k1: p = 2^256 - 2^32 - 977
// For a 512-bit number, reduce mod p
__device__ __forceinline__ void reduce512_PTX(const uint64_t t[8], uint64_t r[4]) {
    // secp256k1 reduction: if input is t[0..7], we need to reduce t[4..7] * 2^256 mod p
    // 2^256 mod p = 2^32 + 977 = 0x1000003D1
    
    uint64_t s[5];
    uint64_t c;
    
    // Multiply high part by 0x1000003D1 and add to low part
    // s = t[0..3] + t[4..7] * 0x1000003D1
    
    const uint64_t k = 0x1000003D1ULL;
    
    // t[4] * k
    uint64_t lo4, hi4;
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo4) : "l"(t[4]), "l"(k));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi4) : "l"(t[4]), "l"(k));
    
    // t[5] * k
    uint64_t lo5, hi5;
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo5) : "l"(t[5]), "l"(k));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi5) : "l"(t[5]), "l"(k));
    
    // t[6] * k
    uint64_t lo6, hi6;
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo6) : "l"(t[6]), "l"(k));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi6) : "l"(t[6]), "l"(k));
    
    // t[7] * k
    uint64_t lo7, hi7;
    asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo7) : "l"(t[7]), "l"(k));
    asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi7) : "l"(t[7]), "l"(k));
    
    // Add to low part
    asm volatile (
        "add.cc.u64 %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.cc.u64 %3, %7, %11;\n\t"
        "addc.u64 %12, 0, 0;\n\t"
        : "=l"(s[0]), "=l"(s[1]), "=l"(s[2]), "=l"(s[3]), "=l"(s[4])
        : "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]),
          "l"(lo4), "l"(lo5 + hi4), "l"(lo6 + hi5), "l"(lo7 + hi6), "l"(hi7)
    );
    
    // Add carries from hi parts
    asm volatile (
        "add.cc.u64 %1, %1, %4;\n\t"
        "addc.cc.u64 %2, %2, %5;\n\t"
        "addc.cc.u64 %3, %3, %6;\n\t"
        "addc.u64 %0, %0, %7;\n\t"
        : "+l"(s[4]), "+l"(s[1]), "+l"(s[2]), "+l"(s[3])
        : "l"(hi4), "l"(hi5), "l"(hi6), "l"(hi7)
    );
    
    // Second reduction if needed (s[4] * k)
    if (s[4]) {
        uint64_t lo, hi;
        asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(s[4]), "l"(k));
        asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(s[4]), "l"(k));
        
        asm volatile (
            "add.cc.u64 %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, %5;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.u64 %3, %3, 0;\n\t"
            : "+l"(s[0]), "+l"(s[1]), "+l"(s[2]), "+l"(s[3])
            : "l"(lo), "l"(hi)
        );
    }
    
    // Final reduction: if s >= p, subtract p
    uint64_t d[4];
    asm volatile (
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.u64 %3, %7, %11;\n\t"
        : "=l"(d[0]), "=l"(d[1]), "=l"(d[2]), "=l"(d[3])
        : "l"(s[0]), "l"(s[1]), "l"(s[2]), "l"(s[3]),
          "l"(0xFFFFFFFEFFFFFC2FULL), "l"(0xFFFFFFFFFFFFFFFFULL), 
          "l"(0xFFFFFFFFFFFFFFFFULL), "l"(0xFFFFFFFFFFFFFFFFULL)
    );
    
    int64_t sign = (int64_t)d[3] >> 63;
    uint64_t mask = (uint64_t)sign;
    
    r[0] = (s[0] & mask) | (d[0] & ~mask);
    r[1] = (s[1] & mask) | (d[1] & ~mask);
    r[2] = (s[2] & mask) | (d[2] & ~mask);
    r[3] = (s[3] & mask) | (d[3] & ~mask);
}

// Ultra-fast field multiplication using PTX
__device__ __forceinline__ void fieldMul_PTX(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t[8];
    mul256x256_PTX(a, b, t);
    reduce512_PTX(t, r);
}

// Ultra-fast field squaring (slightly faster than mul since a=b)
__device__ __forceinline__ void fieldSqr_PTX(const uint64_t a[4], uint64_t r[4]) {
    fieldMul_PTX(a, a, r);
}

// Optimized Fermat inversion: a^(p-2) mod p
// Uses addition chain optimized for secp256k1
__device__ void fieldInv_PTX(const uint64_t a[4], uint64_t r[4]) {
    uint64_t x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4], x88[4], x176[4], x220[4], x223[4];
    uint64_t t[4];
    
    // x2 = a^2
    fieldSqr_PTX(a, x2);
    // x2 = a^3
    fieldMul_PTX(x2, a, x3);
    // x6 = a^6 = (a^3)^2
    fieldSqr_PTX(x3, x6);
    // x9 = a^9 = a^6 * a^3
    fieldMul_PTX(x6, x3, x9);
    // x11 = a^11 = a^9 * a^2
    fieldMul_PTX(x9, x2, x11);
    // x22 = a^22 = (a^11)^2
    fieldSqr_PTX(x11, x22);
    // x44 = a^44 = (a^22)^2
    fieldSqr_PTX(x22, t);
    fieldSqr_PTX(t, x44);
    // x88 = a^88 = (a^44)^2^2
    fieldSqr_PTX(x44, t);
    fieldSqr_PTX(t, t);
    fieldSqr_PTX(t, t);
    fieldSqr_PTX(t, x88);
    // x176 = a^176 = (a^88)^2
    fieldSqr_PTX(x88, t);
    for(int i=0; i<7; i++) fieldSqr_PTX(t, t);
    fieldMul_PTX(t, x88, x176);
    // x220 = a^220 = x176 * x44
    fieldSqr_PTX(x176, t);
    for(int i=0; i<3; i++) fieldSqr_PTX(t, t);
    fieldMul_PTX(t, x44, x220);
    // x223 = a^223 = x220 * x3
    fieldMul_PTX(x220, x3, x223);
    
    // Now compute a^(p-2) using the exponent structure
    // p-2 = 2^256 - 2^32 - 979 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    
    // Start with x223, then 23 squarings, then x22, then 5 squarings, then x9, then 3 squarings, then x11, then 2 squarings, then 1
    for(int i=0; i<4; i++) t[i] = x223[i];
    for(int i=0; i<23; i++) fieldSqr_PTX(t, t);
    fieldMul_PTX(t, x22, t);
    for(int i=0; i<5; i++) fieldSqr_PTX(t, t);
    fieldMul_PTX(t, a, t);
    for(int i=0; i<3; i++) fieldSqr_PTX(t, t);
    fieldMul_PTX(t, x2, t);
    fieldSqr_PTX(t, t);
    fieldMul_PTX(t, a, r);
}
