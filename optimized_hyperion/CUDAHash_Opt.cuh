#ifndef CUDA_HASH_OPT_CUH
#define CUDA_HASH_OPT_CUH

#include <cstdint>

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

#define ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define S0(x) (ROR(x,2)^ROR(x,13)^ROR(x,22))
#define S1(x) (ROR(x,6)^ROR(x,11)^ROR(x,25))
#define s0(x) (ROR(x,7)^ROR(x,18)^((x)>>3))
#define s1(x) (ROR(x,17)^ROR(x,19)^((x)>>10))

// Otimização 1: Passar coordenadas como valores individuais (não array) para evitar memory loads
__device__ __forceinline__ void sha256_opt(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, 
                                            uint8_t parity, uint32_t* out) {
    uint32_t w[64];
    
    // Message schedule (primeira parte - dados de entrada)
    w[0] = ((uint32_t)parity << 24) | ((x3 >> 40) & 0xFFFFFF);
    w[1] = (uint32_t)(x3 >> 8);
    w[2] = ((uint32_t)(x3 & 0xFF) << 24) | ((x2 >> 40) & 0xFFFFFF);
    w[3] = (uint32_t)(x2 >> 8);
    w[4] = ((uint32_t)(x2 & 0xFF) << 24) | ((x1 >> 40) & 0xFFFFFF);
    w[5] = (uint32_t)(x1 >> 8);
    w[6] = ((uint32_t)(x1 & 0xFF) << 24) | ((x0 >> 40) & 0xFFFFFF);
    w[7] = (uint32_t)(x0 >> 8);
    w[8] = ((uint32_t)(x0 & 0xFF) << 24) | 0x800000;
    w[9]=w[10]=w[11]=w[12]=w[13]=w[14]=0;
    w[15] = 264;
    
    // Message schedule expansion - UNROLL COMPLETO
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];
    }
    
    // Compression - UNROLL COMPLETO
    uint32_t a=0x6a09e667, b=0xbb67ae85, c=0x3c6ef372, d=0xa54ff53a;
    uint32_t e=0x510e527f, f=0x9b05688c, g=0x1f83d9ab, h=0x5be0cd19;
    
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + S1(e) + CH(e,f,g) + K_SHA[i] + w[i];
        uint32_t t2 = S0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    
    out[0] = 0x6a09e667+a; out[1] = 0xbb67ae85+b;
    out[2] = 0x3c6ef372+c; out[3] = 0xa54ff53a+d;
    out[4] = 0x510e527f+e; out[5] = 0x9b05688c+f;
    out[6] = 0x1f83d9ab+g; out[7] = 0x5be0cd19+h;
}

#define ROL(x,n) (((x)<<(n))|((x)>>(32-(n))))
#define F(x,y,z) ((x)^(y)^(z))
#define G(x,y,z) (((x)&(y))|(~(x)&(z)))
#define H(x,y,z) (((x)|~(y))^(z))
#define I(x,y,z) (((x)&(z))|((y)&~(z)))
#define J(x,y,z) ((x)^((y)|~(z)))

// Otimização 2: RIPEMD160 com lookup tables inline e unroll total
__device__ __forceinline__ void ripemd160_opt(const uint32_t* sha, uint8_t* out) {
    uint32_t x[16];
    
    // Byte swap
    #pragma unroll
    for(int i=0; i<8; i++) {
        uint32_t v = sha[i];
        x[i] = ((v&0xFF)<<24) | ((v&0xFF00)<<8) | ((v>>8)&0xFF00) | (v>>24);
    }
    x[8]=0x80; x[9]=x[10]=x[11]=x[12]=x[13]=0; x[14]=256; x[15]=0;
    
    uint32_t al=0x67452301, bl=0xefcdab89, cl=0x98badcfe, dl=0x10325476, el=0xc3d2e1f0;
    uint32_t ar=al, br=bl, cr=cl, dr=dl, er=el;
    
    // Left rounds - INLINE COMPLETO
    const int rl[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int sl[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    
    #pragma unroll
    for(int j=0; j<80; j++) {
        uint32_t f, k;
        if(j<16) { f=F(bl,cl,dl); k=0; }
        else if(j<32) { f=G(bl,cl,dl); k=0x5a827999; }
        else if(j<48) { f=H(bl,cl,dl); k=0x6ed9eba1; }
        else if(j<64) { f=I(bl,cl,dl); k=0x8f1bbcdc; }
        else { f=J(bl,cl,dl); k=0xa953fd4e; }
        uint32_t t = ROL(al+f+x[rl[j]]+k, sl[j]) + el;
        al=el; el=dl; dl=ROL(cl,10); cl=bl; bl=t;
    }
    
    // Right rounds
    const int rr[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const int sr[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    
    #pragma unroll
    for(int j=0; j<80; j++) {
        uint32_t f, k;
        if(j<16) { f=J(br,cr,dr); k=0x50a28be6; }
        else if(j<32) { f=I(br,cr,dr); k=0x5c4dd124; }
        else if(j<48) { f=H(br,cr,dr); k=0x6d703ef3; }
        else if(j<64) { f=G(br,cr,dr); k=0x7a6d76e9; }
        else { f=F(br,cr,dr); k=0; }
        uint32_t t = ROL(ar+f+x[rr[j]]+k, sr[j]) + er;
        ar=er; er=dr; dr=ROL(cr,10); cr=br; br=t;
    }
    
    uint32_t h0 = 0xefcdab89U+cl+dr;
    uint32_t h1 = 0x98badcfeU+dl+er;
    uint32_t h2 = 0x10325476U+el+ar;
    uint32_t h3 = 0xc3d2e1f0U+al+br;
    uint32_t h4 = 0x67452301U+bl+cr;
    
    out[0]=h0; out[1]=h0>>8; out[2]=h0>>16; out[3]=h0>>24;
    out[4]=h1; out[5]=h1>>8; out[6]=h1>>16; out[7]=h1>>24;
    out[8]=h2; out[9]=h2>>8; out[10]=h2>>16; out[11]=h2>>24;
    out[12]=h3; out[13]=h3>>8; out[14]=h3>>16; out[15]=h3>>24;
    out[16]=h4; out[17]=h4>>8; out[18]=h4>>16; out[19]=h4>>24;
}

#endif
