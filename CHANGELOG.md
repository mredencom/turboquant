# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2025-07-14

### Added

- Core TurboQuant quantizer with `NewTurboQuant` factory function (dimension, bitWidth, seed)
- 2-bit, 3-bit, and 4-bit vector quantization and dequantization
- Lloyd-Max codebook construction on Beta distribution with configurable grid points and iterations
- Thread-safe codebook caching via `sync.Map` (keyed by dimension and bitWidth)
- Random orthogonal rotation matrix generation via QR decomposition (`gonum/mat`)
- Compact binary serialization and deserialization with bit-packed index storage
- Batch quantization and dequantization with concurrent goroutine execution
- Cosine similarity and compression ratio utility functions
- Comprehensive test suite: unit tests, property-based tests, and example tests
- Benchmark suite covering quantize, dequantize, serialize, codebook build, and rotation matrix generation
- CI pipeline with Go 1.22/1.23/1.24 matrix, race detection, and golangci-lint
- Package-level documentation (`doc.go`) and runnable examples for pkg.go.dev
