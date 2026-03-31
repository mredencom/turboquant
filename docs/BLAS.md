# Performance Tuning: BLAS Backend Configuration

TurboQuant relies on [gonum](https://github.com/gonum/gonum) for linear algebra operations — primarily matrix-vector multiplication in the rotation step (`rotation.go`). By default, gonum uses a **pure Go BLAS implementation** (`gonum/internal/asm`). For large dimensions, linking an optimized native BLAS library can significantly improve performance.

## Default Backend: Pure Go

Out of the box, gonum ships with a pure Go BLAS/LAPACK implementation. No CGO or external libraries are required.

- **Package**: `gonum.org/v1/gonum/blas/gonum` (used automatically)
- **Pros**: Zero dependencies, cross-compiles easily, works everywhere Go runs
- **Cons**: No SIMD vectorization beyond what the Go compiler auto-vectorizes; slower for large matrix operations

For typical embedding dimensions (≤ 512), the pure Go backend is **fast enough** for most use cases. The overhead of CGO calls can actually make native BLAS *slower* for small matrices.

## Benchmark Results (Pure Go Backend)

Measured on Apple M4, `go1.24`, `gonum v0.17.0`:

### Matrix Generation (QR Decomposition)

| Dimension | Time/op | Allocs/op |
|-----------|---------|-----------|
| 64 | 0.39 ms | 17 |
| 128 | 2.6 ms | 18 |
| 256 | 13.5 ms | 219 |
| 512 | 59.1 ms | 1,674 |

### Single-Vector Quantize (includes rotation)

| Dimension | 2-bit | 3-bit | 4-bit |
|-----------|-------|-------|-------|
| 128 | 24.6 µs | 19.0 µs | 19.3 µs |
| 256 | 85.0 µs | 84.6 µs | 92.7 µs |
| 512 | 385.9 µs | 384.2 µs | 387.7 µs |
| 1024 | 1.63 ms | 1.62 ms | 1.63 ms |

### Single-Vector Dequantize (includes inverse rotation)

| Dimension | 2-bit | 3-bit | 4-bit |
|-----------|-------|-------|-------|
| 128 | 10.6 µs | 10.6 µs | 10.6 µs |
| 256 | 47.9 µs | 38.8 µs | 38.9 µs |
| 512 | 140.7 µs | 140.6 µs | 140.8 µs |
| 1024 | 553.0 µs | 544.8 µs | 537.1 µs |

### Batch Quantize (dim=256, 4-bit, concurrent)

| Batch Size | Time/op | Allocs/op |
|------------|---------|-----------|
| 100 | 1.45 ms | 907 |
| 1,000 | 16.8 ms | 9,038 |
| 10,000 | 158.7 ms | 90,072 |

> The matrix-vector multiply (`MulVec`) in `Apply` / `ApplyTranspose` dominates quantize/dequantize time. This is the operation that benefits most from an optimized BLAS backend.

## Linking OpenBLAS

[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) is a widely available, high-performance open-source BLAS library with SIMD optimizations for x86 (AVX/AVX-512) and ARM (NEON).

### 1. Install OpenBLAS

**macOS (Homebrew):**
```bash
brew install openblas
```

**Ubuntu / Debian:**
```bash
sudo apt-get install libopenblas-dev
```

**Fedora / RHEL:**
```bash
sudo dnf install openblas-devel
```

### 2. Use the gonum CGO BLAS Binding

gonum provides a CGO wrapper that delegates to the system BLAS library. To activate it, add a blank import in your application (not in the library itself):

```go
package main

import (
    _ "gonum.org/v1/gonum/blas/cgo"  // Link system BLAS via CGO
    "github.com/mredencom/turboquant"
)
```

The blank import registers the CGO BLAS implementation as the default backend for all gonum operations, including those used by TurboQuant.

### 3. Build with CGO Enabled

```bash
CGO_ENABLED=1 go build ./...
```

On macOS with Homebrew OpenBLAS, you may need to set pkg-config or linker flags:

```bash
export CGO_LDFLAGS="-L$(brew --prefix openblas)/lib -lopenblas"
export CGO_CFLAGS="-I$(brew --prefix openblas)/include"
CGO_ENABLED=1 go build ./...
```

### 4. Verify the Backend

You can confirm the CGO backend is active by checking that `gonum.org/v1/gonum/blas/cgo` is imported. If the import is missing or CGO is disabled, gonum silently falls back to the pure Go implementation.

## Linking Intel MKL

[Intel oneAPI Math Kernel Library (MKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) provides highly optimized BLAS routines for Intel CPUs, with automatic dispatch to AVX-512 on supported hardware.

### 1. Install MKL

**Ubuntu / Debian (via Intel APT repo):**
```bash
# Add Intel repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-mkl-devel
```

**macOS:** MKL is available via the Intel oneAPI installer but is less commonly used on Apple Silicon. OpenBLAS with ARM NEON is the recommended choice on Apple Silicon Macs.

### 2. Link MKL as the BLAS Provider

The same `gonum/blas/cgo` import works with MKL — you just need to point the linker at MKL instead of OpenBLAS:

```bash
source /opt/intel/oneapi/setvars.sh  # Set up MKL environment
export CGO_LDFLAGS="-lmkl_rt"
CGO_ENABLED=1 go build ./...
```

For static linking or specific MKL configurations (LP64 vs ILP64, sequential vs threaded), refer to the [Intel MKL Link Line Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html).

## Expected Performance Improvements

The primary operation that benefits from native BLAS is **DGEMV** (double-precision matrix-vector multiply), used in `Matrix.Apply` and `Matrix.ApplyTranspose`.

Based on gonum's own benchmarks and community reports:

| Dimension | Pure Go | OpenBLAS (est.) | MKL (est.) | Speedup |
|-----------|---------|-----------------|------------|---------|
| 128 | ~19 µs | ~8–12 µs | ~6–10 µs | 1.5–3× |
| 256 | ~85 µs | ~25–40 µs | ~20–35 µs | 2–4× |
| 512 | ~385 µs | ~80–150 µs | ~60–120 µs | 2.5–6× |
| 1024 | ~1.6 ms | ~250–500 µs | ~200–400 µs | 3–8× |

> **Note:** These are estimates based on typical BLAS speedups for DGEMV. Actual results depend on hardware, matrix layout, and cache effects. The speedup grows with dimension because native BLAS can exploit SIMD and cache-aware blocking strategies that the pure Go implementation cannot.

### When Native BLAS Helps Most

- **Dimension ≥ 512**: Clear wins from SIMD vectorization and cache optimization
- **Batch operations**: The per-vector rotation cost dominates, so BLAS speedup multiplies across the batch
- **Matrix generation**: QR decomposition for `NewRandomOrthogonalMatrix` also benefits (one-time cost)

### When Pure Go Is Fine

- **Dimension < 256**: CGO call overhead can negate BLAS gains for small matrices
- **Cross-compilation needed**: CGO complicates cross-compilation; pure Go "just works"
- **Deployment simplicity**: No system library dependencies to manage

## Recommendation

For most users with embedding dimensions ≤ 512, the **pure Go backend is sufficient**. The quantize/dequantize latency is dominated by the rotation step, but at these dimensions it's already in the microsecond range.

If you're processing high-dimensional vectors (≥ 1024) or need maximum throughput for large batches, linking OpenBLAS is the easiest way to get a 2–6× speedup on the rotation step. MKL offers the best performance on Intel hardware but adds deployment complexity.

```
Dimension ≤ 256  → Pure Go (default, no action needed)
Dimension 256–512 → Consider OpenBLAS if latency-sensitive
Dimension ≥ 1024  → Recommend OpenBLAS or MKL
```
