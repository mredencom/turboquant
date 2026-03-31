package turboquant

import (
	"fmt"
	"math"
)

// BetaPDF computes the probability density of the Beta(alpha, beta) distribution at x.
// Uses log-space computation via math.Lgamma to avoid numerical overflow.
// Returns 0.0 for x outside the open interval (0, 1).
func BetaPDF(x, alpha, beta float64) float64 {
	if x <= 0 || x >= 1 {
		return 0.0
	}

	// log of the Beta function: B(α,β) = Γ(α)Γ(β)/Γ(α+β)
	lgA, _ := math.Lgamma(alpha)
	lgB, _ := math.Lgamma(beta)
	lgAB, _ := math.Lgamma(alpha + beta)
	logBeta := lgA + lgB - lgAB

	// log PDF = (α-1)*log(x) + (β-1)*log(1-x) - logB(α,β)
	logPDF := (alpha-1)*math.Log(x) + (beta-1)*math.Log(1-x) - logBeta

	return math.Exp(logPDF)
}

// CosineSimilarity computes the cosine similarity between two float32 vectors.
// Returns a float64 value in [-1, 1].
// Returns an error if the vectors have different dimensions.
// Returns 0.0 if either vector is a zero vector.
func CosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("dimension mismatch: len(a)=%d, len(b)=%d", len(a), len(b))
	}
	if len(a) == 0 {
		return 0.0, nil
	}

	var dot, normA, normB float64
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	if normA == 0 || normB == 0 {
		return 0.0, nil
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

// CompressionRatio computes the theoretical compression ratio for a given dimension and bit width.
// Formula: (dimension * 32) / (32 + dimension * bitWidth)
// Original size: dimension * 32 bits (one float32 per element).
// Compressed size: 32 bits (float32 norm) + dimension * bitWidth bits.
func CompressionRatio(dimension, bitWidth int) float64 {
	return float64(dimension*32) / float64(32+dimension*bitWidth)
}
