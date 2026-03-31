// Package turboquant implements the TurboQuant online vector quantization
// algorithm from "TurboQuant: Online Vector Quantization" (arXiv:2504.19874)
// by Google Research.
//
// TurboQuant compresses float32 vectors into compact low-bit representations
// using a two-step approach: random orthogonal rotation followed by Lloyd-Max
// scalar quantization on a Beta distribution. The algorithm is data-oblivious,
// meaning no training data is needed — codebooks are derived analytically from
// the statistical properties of uniformly distributed unit-sphere vectors.
//
// # Algorithm
//
// After rotating a normalized vector by a random orthogonal matrix, each
// coordinate is approximately distributed as Beta((d-1)/2, (d-1)/2) where d
// is the vector dimension. A Lloyd-Max optimal scalar quantizer is pre-computed
// on this distribution, and each rotated coordinate is independently quantized
// by looking up the nearest centroid. Dequantization reverses the process:
// centroid lookup, inverse rotation (transpose), and norm rescaling.
//
// # Supported Bit Widths
//
// The SDK supports 2-bit, 3-bit, and 4-bit quantization. Higher bit widths
// yield better reconstruction quality (higher cosine similarity) at the cost
// of larger compressed size. For vectors with dimension ≥ 64, typical cosine
// similarities are ≥ 0.90 (2-bit), ≥ 0.96 (3-bit), and ≥ 0.99 (4-bit).
//
// # Usage
//
// The main entry point is [NewTurboQuant], which builds (or retrieves from
// cache) the Lloyd-Max codebook and generates the rotation matrix:
//
//	tq, err := turboquant.NewTurboQuant(128, 4, 42) // dim=128, 4-bit, seed=42
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Quantize a vector
//	qv, err := tq.Quantize(vec)
//
//	// Serialize to compact binary
//	data, err := tq.Serialize(qv)
//
//	// Deserialize and dequantize
//	qv2, err := tq.Deserialize(data)
//	restored, err := tq.Dequantize(qv2)
//
// Batch operations ([TurboQuant.QuantizeBatch], [TurboQuant.DequantizeBatch])
// process multiple vectors concurrently using goroutines.
//
// Codebooks are cached globally by (dimension, bitWidth), so creating multiple
// TurboQuant instances with the same parameters reuses the codebook.
package turboquant
