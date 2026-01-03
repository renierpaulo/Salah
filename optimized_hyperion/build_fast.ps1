param(
  [string]$Arch = "sm_86",
  [int]$MaxRRegCount = 96
)

$ErrorActionPreference = "Stop"

nvcc -O3 -lineinfo -Xptxas=-v --use_fast_math `
  -arch=$Arch `
  -maxrregcount=$MaxRRegCount `
  -DPROFILER_ECC_ONLY=1 `
  -DECC_ONLY_PERSIST_JACOBIAN=0 `
  -I. `
  Hyperion_Profiler_Genius.cu `
  -o Hyperion_Profiler_Genius_fast.exe

Write-Host "Built: .\Hyperion_Profiler_Genius_fast.exe (ARCH=$Arch, MAXRREG=$MaxRRegCount)"
