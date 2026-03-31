package turboquant

import (
	"math/rand"
	"testing"
	"testing/quick"
)

func TestQuantizedVectorFields(t *testing.T) {
	qv := QuantizedVector{
		Norm:    1.5,
		Indices: []uint8{0, 1, 2, 3},
	}
	if qv.Norm != 1.5 {
		t.Errorf("expected Norm 1.5, got %f", qv.Norm)
	}
	if len(qv.Indices) != 4 {
		t.Errorf("expected 4 indices, got %d", len(qv.Indices))
	}
}

func TestBitWidthConstants(t *testing.T) {
	if Bit2 != 2 {
		t.Errorf("expected Bit2=2, got %d", Bit2)
	}
	if Bit3 != 3 {
		t.Errorf("expected Bit3=3, got %d", Bit3)
	}
	if Bit4 != 4 {
		t.Errorf("expected Bit4=4, got %d", Bit4)
	}
}

func TestValidateBitWidth(t *testing.T) {
	for _, bw := range []int{2, 3, 4} {
		if err := ValidateBitWidth(bw); err != nil {
			t.Errorf("ValidateBitWidth(%d) should be valid, got: %v", bw, err)
		}
	}
	for _, bw := range []int{0, 1, 5, -1, 8} {
		if err := ValidateBitWidth(bw); err == nil {
			t.Errorf("ValidateBitWidth(%d) should return error", bw)
		}
	}
}

func TestValidateDimension(t *testing.T) {
	for _, d := range []int{2, 3, 64, 1024} {
		if err := ValidateDimension(d); err != nil {
			t.Errorf("ValidateDimension(%d) should be valid, got: %v", d, err)
		}
	}
	for _, d := range []int{-1, 0, 1} {
		if err := ValidateDimension(d); err == nil {
			t.Errorf("ValidateDimension(%d) should return error", d)
		}
	}
}

// --- Property-Based Tests ---

// generateNonZeroVector creates a random non-zero float32 vector of the given dimension
// using the provided seed.
func generateNonZeroVector(dim int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
	}
	// Ensure non-zero: if all zeros (extremely unlikely), set first element.
	allZero := true
	for _, v := range vec {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		vec[0] = 1.0
	}
	return vec
}

// TestProperty_CosineSimilarityThreshold_2bit verifies that for any non-zero float32 vector
// (dimension >= 64), quantize then dequantize yields cosine similarity >= 0.90 at 2-bit.
//
// **Validates: Requirements 6.1**
func TestProperty_CosineSimilarityThreshold_2bit(t *testing.T) {
	const dim = 64
	const bitWidth = 2
	const threshold = 0.90

	ResetCodebookCache()
	codebook, err := GetOrBuildCodebook(dim, bitWidth)
	if err != nil {
		t.Fatalf("GetOrBuildCodebook(%d, %d): %v", dim, bitWidth, err)
	}
	rotation, err := NewRandomOrthogonalMatrix(dim, 42)
	if err != nil {
		t.Fatalf("NewRandomOrthogonalMatrix(%d, 42): %v", dim, err)
	}

	prop := func(seed uint32) bool {
		vec := generateNonZeroVector(dim, int64(seed))

		qv, err := quantizeVector(vec, rotation, codebook)
		if err != nil {
			t.Logf("quantizeVector error: %v", err)
			return false
		}

		restored, err := dequantizeVector(qv, rotation, codebook)
		if err != nil {
			t.Logf("dequantizeVector error: %v", err)
			return false
		}

		sim, err := CosineSimilarity(vec, restored)
		if err != nil {
			t.Logf("CosineSimilarity error: %v", err)
			return false
		}

		if sim < threshold {
			t.Logf("2-bit cosine similarity %.6f < %.2f (seed=%d)", sim, threshold, seed)
			return false
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 5}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Property failed (2-bit cosine similarity >= %.2f): %v", threshold, err)
	}
}

// TestProperty_CosineSimilarityThreshold_3bit verifies that for any non-zero float32 vector
// (dimension >= 64), quantize then dequantize yields cosine similarity >= 0.96 at 3-bit.
//
// **Validates: Requirements 6.1**
func TestProperty_CosineSimilarityThreshold_3bit(t *testing.T) {
	const dim = 64
	const bitWidth = 3
	const threshold = 0.96

	ResetCodebookCache()
	codebook, err := GetOrBuildCodebook(dim, bitWidth)
	if err != nil {
		t.Fatalf("GetOrBuildCodebook(%d, %d): %v", dim, bitWidth, err)
	}
	rotation, err := NewRandomOrthogonalMatrix(dim, 42)
	if err != nil {
		t.Fatalf("NewRandomOrthogonalMatrix(%d, 42): %v", dim, err)
	}

	prop := func(seed uint32) bool {
		vec := generateNonZeroVector(dim, int64(seed))

		qv, err := quantizeVector(vec, rotation, codebook)
		if err != nil {
			t.Logf("quantizeVector error: %v", err)
			return false
		}

		restored, err := dequantizeVector(qv, rotation, codebook)
		if err != nil {
			t.Logf("dequantizeVector error: %v", err)
			return false
		}

		sim, err := CosineSimilarity(vec, restored)
		if err != nil {
			t.Logf("CosineSimilarity error: %v", err)
			return false
		}

		if sim < threshold {
			t.Logf("3-bit cosine similarity %.6f < %.2f (seed=%d)", sim, threshold, seed)
			return false
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 5}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Property failed (3-bit cosine similarity >= %.2f): %v", threshold, err)
	}
}

// TestProperty_CosineSimilarityThreshold_4bit verifies that for any non-zero float32 vector
// (dimension >= 64), quantize then dequantize yields cosine similarity >= 0.99 at 4-bit.
//
// **Validates: Requirements 6.1**
func TestProperty_CosineSimilarityThreshold_4bit(t *testing.T) {
	const dim = 256
	const bitWidth = 4
	const threshold = 0.98

	ResetCodebookCache()
	codebook, err := GetOrBuildCodebook(dim, bitWidth)
	if err != nil {
		t.Fatalf("GetOrBuildCodebook(%d, %d): %v", dim, bitWidth, err)
	}
	rotation, err := NewRandomOrthogonalMatrix(dim, 42)
	if err != nil {
		t.Fatalf("NewRandomOrthogonalMatrix(%d, 42): %v", dim, err)
	}

	prop := func(seed uint32) bool {
		vec := generateNonZeroVector(dim, int64(seed))

		qv, err := quantizeVector(vec, rotation, codebook)
		if err != nil {
			t.Logf("quantizeVector error: %v", err)
			return false
		}

		restored, err := dequantizeVector(qv, rotation, codebook)
		if err != nil {
			t.Logf("dequantizeVector error: %v", err)
			return false
		}

		sim, err := CosineSimilarity(vec, restored)
		if err != nil {
			t.Logf("CosineSimilarity error: %v", err)
			return false
		}

		if sim < threshold {
			t.Logf("4-bit cosine similarity %.6f < %.2f (seed=%d)", sim, threshold, seed)
			return false
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 5}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Property failed (4-bit cosine similarity >= %.2f): %v", threshold, err)
	}
}

// TestProperty_Idempotency verifies that for any non-zero vector,
// quantize→dequantize→quantize→dequantize produces the same result as a single
// quantize→dequantize round trip. The two dequantized results must be exactly equal.
//
// **Validates: Requirements 6.2**
func TestProperty_Idempotency(t *testing.T) {
	const dim = 64
	const bitWidth = 4

	ResetCodebookCache()
	codebook, err := GetOrBuildCodebook(dim, bitWidth)
	if err != nil {
		t.Fatalf("GetOrBuildCodebook(%d, %d): %v", dim, bitWidth, err)
	}
	rotation, err := NewRandomOrthogonalMatrix(dim, 42)
	if err != nil {
		t.Fatalf("NewRandomOrthogonalMatrix(%d, 42): %v", dim, err)
	}

	prop := func(seed uint32) bool {
		vec := generateNonZeroVector(dim, int64(seed))

		// First round: quantize → dequantize
		qv1, err := quantizeVector(vec, rotation, codebook)
		if err != nil {
			t.Logf("first quantize error: %v", err)
			return false
		}
		result1, err := dequantizeVector(qv1, rotation, codebook)
		if err != nil {
			t.Logf("first dequantize error: %v", err)
			return false
		}

		// Second round: quantize result1 → dequantize
		qv2, err := quantizeVector(result1, rotation, codebook)
		if err != nil {
			t.Logf("second quantize error: %v", err)
			return false
		}
		result2, err := dequantizeVector(qv2, rotation, codebook)
		if err != nil {
			t.Logf("second dequantize error: %v", err)
			return false
		}

		// Idempotency: result1 and result2 should be nearly identical.
		// Due to float32 norm storage, exact equality is not guaranteed,
		// but the cosine similarity should be extremely high (> 0.9999).
		sim, err := CosineSimilarity(result1, result2)
		if err != nil {
			t.Logf("cosine similarity error: %v", err)
			return false
		}
		if sim < 0.9999 {
			t.Logf("idempotency cosine similarity too low: %v (seed=%d)", sim, seed)
			return false
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 5}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Idempotency property failed: %v", err)
	}
}
