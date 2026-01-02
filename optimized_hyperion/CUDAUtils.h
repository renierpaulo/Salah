#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>
#include <cstdint>
#include <cstring>

bool parseHash160(const char* hex, uint8_t* h) {
    if (strlen(hex) != 40) return false;
    for (int i = 0; i < 20; i++) { 
        unsigned int b; 
        if (sscanf(hex+i*2, "%2x", &b) != 1) return false; 
        h[i] = b; 
    }
    return true;
}

bool parseHex256(const char* hex, uint64_t* limbs) {
    limbs[0]=limbs[1]=limbs[2]=limbs[3]=0;
    int len = strlen(hex); 
    if (len > 64) return false;
    char padded[65] = {0}; 
    memset(padded, '0', 64-len); 
    strcpy(padded+64-len, hex);
    for (int i=0; i<4; i++) { 
        char chunk[17]={0}; 
        strncpy(chunk, padded+i*16, 16); 
        limbs[3-i] = strtoull(chunk, NULL, 16); 
    }
    return true;
}

void printKey256(const char* label, uint64_t* k) {
    printf("%s%016llx%016llx%016llx%016llx\n", 
           label,
           (unsigned long long)k[3], 
           (unsigned long long)k[2], 
           (unsigned long long)k[1], 
           (unsigned long long)k[0]);
}

#endif
