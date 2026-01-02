#ifndef CUDA_MATH_JACOBIAN_FIXED_H
#define CUDA_MATH_JACOBIAN_FIXED_H

#include "CUDAMath.h"

/*
 * JACOBIAN COORDINATES - SIMPLIFIED VERSION
 * Uses existing CUDAMath.h functions (fieldMul, fieldSqr, fieldInv, fieldSub, fieldAdd)
 */

struct ECPointJ {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    bool infinity;
};

// Convert affine to Jacobian
__device__ __forceinline__ void affineToJacobian(const ECPointA& P, ECPointJ& J) {
    if (P.infinity) {
        J.X[0]=J.X[1]=J.X[2]=J.X[3]=0;
        J.Y[0]=J.Y[1]=J.Y[2]=J.Y[3]=0;
        J.Z[0]=J.Z[1]=J.Z[2]=J.Z[3]=0;
        J.infinity = true;
        return;
    }
    
    for(int i=0; i<4; i++) {
        J.X[i] = P.X[i];
        J.Y[i] = P.Y[i];
    }
    J.Z[0] = 1;
    J.Z[1] = J.Z[2] = J.Z[3] = 0;
    J.infinity = false;
}

// Convert Jacobian to affine
__device__ __forceinline__ void jacobianToAffine(const ECPointJ& J, ECPointA& P) {
    if (J.infinity) {
        P.infinity = true;
        return;
    }
    
    uint64_t invZ[4], invZ2[4], invZ3[4];
    fieldInv(J.Z, invZ);
    fieldSqr(invZ, invZ2);
    fieldMul(invZ, invZ2, invZ3);
    
    fieldMul(J.X, invZ2, P.X);
    fieldMul(J.Y, invZ3, P.Y);
    P.infinity = false;
}

// Point addition in Jacobian: P + Q (Q in affine)
// This is the optimized version - NO inversion needed!
__device__ void pointAddJacobianMixed(const ECPointJ& P, const ECPointA& Q, ECPointJ& R) {
    if (P.infinity) {
        affineToJacobian(Q, R);
        return;
    }
    if (Q.infinity) {
        R = P;
        return;
    }
    
    uint64_t Z2[4], U2[4], S2[4], H[4], HH[4], I[4], J[4], r[4], V[4], T[4];
    
    // Z2 = Z1²
    fieldSqr(P.Z, Z2);
    
    // U2 = X2*Z1²
    fieldMul(Q.X, Z2, U2);
    
    // S2 = Y2*Z1³
    fieldMul(P.Z, Z2, T);
    fieldMul(Q.Y, T, S2);
    
    // H = U2 - X1
    fieldSub(U2, P.X, H);
    
    // Check if same point
    bool same_x = true;
    for(int i=0; i<4; i++) if(H[i] != 0) { same_x = false; break; }
    
    if (same_x) {
        fieldSub(S2, P.Y, T);
        bool same_y = true;
        for(int i=0; i<4; i++) if(T[i] != 0) { same_y = false; break; }
        
        if (same_y) {
            // Same point - would need doubling, just return P for now
            R = P;
            return;
        } else {
            // Inverse points
            R.infinity = true;
            return;
        }
    }
    
    // r = 2*(S2 - Y1)
    fieldSub(S2, P.Y, r);
    fieldAdd(r, r, r);
    
    // I = (2*H)²
    fieldAdd(H, H, I);
    fieldSqr(I, I);
    
    // J = H*I
    fieldMul(H, I, J);
    
    // V = X1*I
    fieldMul(P.X, I, V);
    
    // X3 = r² - J - 2*V
    fieldSqr(r, R.X);
    fieldSub(R.X, J, R.X);
    fieldSub(R.X, V, R.X);
    fieldSub(R.X, V, R.X);
    
    // Y3 = r*(V - X3) - 2*Y1*J
    fieldSub(V, R.X, T);
    fieldMul(r, T, R.Y);
    fieldMul(P.Y, J, T);
    fieldAdd(T, T, T);
    fieldSub(R.Y, T, R.Y);
    
    // Z3 = 2*Z1*H
    fieldMul(P.Z, H, R.Z);
    fieldAdd(R.Z, R.Z, R.Z);
    
    R.infinity = false;
}

#endif
