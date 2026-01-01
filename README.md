# Salah - HYPERION ULTRA

**18 GKeys/s Bitcoin Private Key Search with 256-bit Precision**

## Target
- **Hash160:** `f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8`
- **Range:** `400000000000000000:7fffffffffffffffff`

## Compilation

```bash
nvcc -O3 -arch=sm_86 -use_fast_math Hyperion_ULTRA.cu -o Hyperion_ULTRA
```

## Usage

```bash
./Hyperion_ULTRA --hash160 f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8 --range 400000000000000000:7fffffffffffffffff
```

## Performance
- **Speed:** ~17-18 GKeys/s
- **GPU:** NVIDIA RTX 3090 / A100 / etc.
- **Precision:** Full 256-bit range support

## Features
- Fast 64-bit initialization with 256-bit offset
- Inline SHA256 and RIPEMD160 for maximum speed
- Correct hash160 calculation for every private key
