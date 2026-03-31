# TurboQuant

A Go library implementing the TurboQuant online vector quantization algorithm ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)). It compresses float32 vectors to 2/3/4-bit representations using random orthogonal rotation and Lloyd-Max scalar quantization on the Beta distribution — no training data required.

## Features

- 2-bit, 3-bit, and 4-bit quantization
- Data-oblivious: works without training data
- Concurrent batch quantize/dequantize via goroutines
- Compact binary serialization (bit-packed)
- Codebook auto-caching (thread-safe)
- Deterministic: same seed → same results

## Install

```bash
go get github.com/mredencom/turboquant@latest
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/mredencom/turboquant"
)

func main() {
    // Create a 4-bit quantizer for 128-dimensional vectors
    tq, err := turboquant.NewTurboQuant(128, turboquant.Bit4, 42)
    if err != nil {
        panic(err)
    }

    // Quantize
    vec := make([]float32, 128)
    for i := range vec {
        vec[i] = float32(i) * 0.01
    }
    qv, _ := tq.Quantize(vec)

    // Serialize → Deserialize
    data, _ := tq.Serialize(qv)
    qv2, _ := tq.Deserialize(data)

    // Dequantize
    restored, _ := tq.Dequantize(qv2)

    // Check quality
    sim, _ := turboquant.CosineSimilarity(vec, restored)
    fmt.Printf("Cosine similarity: %.4f\n", sim)
    fmt.Printf("Compression ratio: %.1fx\n", tq.CompressionRatio())
}
```

## API

| Method | Description |
|---|---|
| `NewTurboQuant(dimension, bitWidth, seed)` | Create a quantizer instance |
| `Quantize(vec)` | Quantize a single float32 vector |
| `Dequantize(qv)` | Reconstruct a float32 vector |
| `QuantizeBatch(vecs)` | Batch quantize (concurrent) |
| `DequantizeBatch(qvs)` | Batch dequantize (concurrent) |
| `Serialize(qv)` | Serialize to compact binary |
| `Deserialize(data)` | Deserialize from binary |
| `CompressionRatio()` | Get theoretical compression ratio |
| `CosineSimilarity(a, b)` | Compute cosine similarity between two vectors |

## How It Works

1. Compute L2 norm and normalize the input vector
2. Apply a random orthogonal rotation (QR decomposition)
3. Quantize each rotated coordinate using a Lloyd-Max codebook optimized for the Beta distribution
4. Store the norm (float32) + quantized indices (bit-packed)

Dequantization reverses the process: look up centroids → inverse rotation → scale by norm.

## Project Structure

```
turboquant.go      Public API: NewTurboQuant, Quantize, Dequantize, Batch, Serialize
codebook.go        Lloyd-Max codebook builder with cache
rotation.go        Random orthogonal matrix via QR decomposition
quantize.go        Core quantize/dequantize logic
serialize.go       Bit-packed binary serialization
math_utils.go      Beta PDF, cosine similarity, compression ratio
convert.go         Type conversion helpers (float64, int, byte, string → float32)
```

## Testing

```bash
go test -v ./...
```

50 tests including property-based tests for correctness properties:
- Codebook centroid count = 2^bitWidth
- Rotation matrix orthogonality (R^T·R ≈ I)
- Rotation reproducibility (same seed → same matrix)
- Quantize-dequantize cosine similarity thresholds
- Serialization round-trip consistency

## License

MIT
