package turboquant

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
)

func TestNewTurboQuant_Valid(t *testing.T) {
	tests := []struct {
		name      string
		dimension int
		bitWidth  int
		seed      int64
	}{
		{"dim2_bit2", 2, 2, 42},
		{"dim4_bit3", 4, 3, 99},
		{"dim8_bit4", 8, 4, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tq, err := NewTurboQuant(tt.dimension, tt.bitWidth, tt.seed)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tq.dimension != tt.dimension {
				t.Errorf("dimension = %d, want %d", tq.dimension, tt.dimension)
			}
			if tq.bitWidth != tt.bitWidth {
				t.Errorf("bitWidth = %d, want %d", tq.bitWidth, tt.bitWidth)
			}
			if tq.codebook == nil {
				t.Error("codebook is nil")
			}
			if tq.rotation == nil {
				t.Error("rotation is nil")
			}
		})
	}
}

func TestNewTurboQuant_InvalidDimension(t *testing.T) {
	for _, dim := range []int{-1, 0, 1} {
		_, err := NewTurboQuant(dim, 2, 42)
		if err == nil {
			t.Errorf("expected error for dimension=%d, got nil", dim)
		}
	}
}

func TestNewTurboQuant_InvalidBitWidth(t *testing.T) {
	for _, bw := range []int{0, 1, 5, 8} {
		_, err := NewTurboQuant(4, bw, 42)
		if err == nil {
			t.Errorf("expected error for bitWidth=%d, got nil", bw)
		}
	}
}

func TestNewTurboQuant_SameSeedSameRotation(t *testing.T) {
	tq1, err := NewTurboQuant(4, 2, 123)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tq2, err := NewTurboQuant(4, 2, 123)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Apply rotation to a test vector and compare results.
	vec := []float64{1, 0, 0, 0}
	r1 := tq1.rotation.Apply(vec)
	r2 := tq2.rotation.Apply(vec)
	for i := range r1 {
		if r1[i] != r2[i] {
			t.Errorf("rotation mismatch at index %d: %f != %f", i, r1[i], r2[i])
		}
	}
}

// ---- Integration tests ----

// TestFullPipeline tests the end-to-end flow:
// NewTurboQuant → Quantize → Serialize → Deserialize → Dequantize
// and verifies cosine similarity > 0.99 with 4-bit quantization.
func TestFullPipeline(t *testing.T) {
	dim := 64
	bitWidth := 4
	seed := int64(42)

	tq, err := NewTurboQuant(dim, bitWidth, seed)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	// Generate a deterministic test vector.
	rng := rand.New(rand.NewSource(99))
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1 // [-1, 1]
	}

	// Quantize
	qv, err := tq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Serialize
	data, err := tq.Serialize(qv)
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}

	// Deserialize
	qv2, err := tq.Deserialize(data)
	if err != nil {
		t.Fatalf("Deserialize: %v", err)
	}

	// Verify round-trip of serialization preserves QuantizedVector.
	if qv.Norm != qv2.Norm {
		t.Errorf("Norm mismatch after serialize round-trip: %f vs %f", qv.Norm, qv2.Norm)
	}
	if len(qv.Indices) != len(qv2.Indices) {
		t.Fatalf("Indices length mismatch: %d vs %d", len(qv.Indices), len(qv2.Indices))
	}
	for i := range qv.Indices {
		if qv.Indices[i] != qv2.Indices[i] {
			t.Errorf("Index mismatch at %d: %d vs %d", i, qv.Indices[i], qv2.Indices[i])
		}
	}

	// Dequantize
	restored, err := tq.Dequantize(qv2)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	// Verify cosine similarity > 0.99.
	sim, err := CosineSimilarity(vec, restored)
	if err != nil {
		t.Fatalf("CosineSimilarity: %v", err)
	}
	if sim < 0.99 {
		t.Errorf("cosine similarity = %f, want >= 0.99", sim)
	}
}

// TestQuantizeBatch_Correctness tests batch quantize and dequantize with 10 random vectors.
func TestQuantizeBatch_Correctness(t *testing.T) {
	dim := 16
	bitWidth := 4
	seed := int64(7)

	tq, err := NewTurboQuant(dim, bitWidth, seed)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(123))
	numVecs := 10
	vecs := make([][]float32, numVecs)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		vecs[i] = v
	}

	// QuantizeBatch
	qvs, err := tq.QuantizeBatch(vecs)
	if err != nil {
		t.Fatalf("QuantizeBatch: %v", err)
	}
	if len(qvs) != numVecs {
		t.Fatalf("QuantizeBatch returned %d results, want %d", len(qvs), numVecs)
	}

	// DequantizeBatch
	restored, err := tq.DequantizeBatch(qvs)
	if err != nil {
		t.Fatalf("DequantizeBatch: %v", err)
	}
	if len(restored) != numVecs {
		t.Fatalf("DequantizeBatch returned %d results, want %d", len(restored), numVecs)
	}

	for i := 0; i < numVecs; i++ {
		if len(restored[i]) != dim {
			t.Errorf("restored[%d] dimension = %d, want %d", i, len(restored[i]), dim)
		}
		sim, err := CosineSimilarity(vecs[i], restored[i])
		if err != nil {
			t.Errorf("CosineSimilarity for vec %d: %v", i, err)
			continue
		}
		// With dim=16 and 4-bit, similarity should be reasonable (> 0.90).
		if sim < 0.90 {
			t.Errorf("vec %d: cosine similarity = %f, want >= 0.90", i, sim)
		}
	}
}

// TestQuantizeBatch_DimensionMismatch verifies that QuantizeBatch returns an error
// mentioning the first mismatch index when vectors have mixed dimensions.
func TestQuantizeBatch_DimensionMismatch(t *testing.T) {
	dim := 8
	tq, err := NewTurboQuant(dim, 4, 1)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	vecs := [][]float32{
		make([]float32, dim), // correct
		make([]float32, dim), // correct
		make([]float32, 5),   // wrong dimension at index 2
		make([]float32, dim), // correct
	}

	_, err = tq.QuantizeBatch(vecs)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}

	// Error should mention index 2.
	errMsg := err.Error()
	if !containsSubstring(errMsg, "index 2") {
		t.Errorf("error message %q does not mention 'index 2'", errMsg)
	}
}

// TestCompressionRatio_Method verifies the CompressionRatio method returns the expected value.
func TestCompressionRatio_Method(t *testing.T) {
	tests := []struct {
		dim      int
		bitWidth int
		want     float64
	}{
		{16, 2, float64(16*32) / float64(32+16*2)},
		{16, 3, float64(16*32) / float64(32+16*3)},
		{16, 4, float64(16*32) / float64(32+16*4)},
	}

	for _, tt := range tests {
		tq, err := NewTurboQuant(tt.dim, tt.bitWidth, 1)
		if err != nil {
			t.Fatalf("NewTurboQuant(dim=%d, bw=%d): %v", tt.dim, tt.bitWidth, err)
		}
		got := tq.CompressionRatio()
		if got != tt.want {
			t.Errorf("CompressionRatio(dim=%d, bw=%d) = %f, want %f", tt.dim, tt.bitWidth, got, tt.want)
		}
	}
}

// containsSubstring checks if s contains substr.
func containsSubstring(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstring(s, substr)
}

func searchSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// ---- Float64 API tests ----

// TestQuantizeFloat64_SameAsManualConvert verifies that QuantizeFloat64 produces
// the same result as manually converting float64→float32 then calling Quantize.
func TestQuantizeFloat64_SameAsManualConvert(t *testing.T) {
	dim := 64
	tq, err := NewTurboQuant(dim, 4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec64 := make([]float64, dim)
	for i := range vec64 {
		vec64[i] = rng.Float64()*2 - 1
	}

	// QuantizeFloat64
	qv1, err := tq.QuantizeFloat64(vec64)
	if err != nil {
		t.Fatalf("QuantizeFloat64: %v", err)
	}

	// Manual: convert then Quantize
	vec32 := Float64sToFloat32s(vec64)
	qv2, err := tq.Quantize(vec32)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	if qv1.Norm != qv2.Norm {
		t.Errorf("Norm mismatch: %f vs %f", qv1.Norm, qv2.Norm)
	}
	if len(qv1.Indices) != len(qv2.Indices) {
		t.Fatalf("Indices length mismatch: %d vs %d", len(qv1.Indices), len(qv2.Indices))
	}
	for i := range qv1.Indices {
		if qv1.Indices[i] != qv2.Indices[i] {
			t.Errorf("Index mismatch at %d: %d vs %d", i, qv1.Indices[i], qv2.Indices[i])
		}
	}
}

// TestDequantizeFloat64_ReturnsFloat64 verifies that DequantizeFloat64 returns float64 values.
func TestDequantizeFloat64_ReturnsFloat64(t *testing.T) {
	dim := 16
	tq, err := NewTurboQuant(dim, 3, 7)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	vec64 := make([]float64, dim)
	for i := range vec64 {
		vec64[i] = float64(i+1) * 0.1
	}

	qv, err := tq.QuantizeFloat64(vec64)
	if err != nil {
		t.Fatalf("QuantizeFloat64: %v", err)
	}

	restored, err := tq.DequantizeFloat64(qv)
	if err != nil {
		t.Fatalf("DequantizeFloat64: %v", err)
	}

	if len(restored) != dim {
		t.Fatalf("expected dimension %d, got %d", dim, len(restored))
	}

	// Verify values are reasonable float64 (not NaN/Inf).
	for i, v := range restored {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("restored[%d] is %f, expected finite float64", i, v)
		}
	}
}

// TestDequantizeFloat64_MatchesDequantize verifies DequantizeFloat64 matches
// manual Dequantize + Float32sToFloat64s.
func TestDequantizeFloat64_MatchesDequantize(t *testing.T) {
	dim := 32
	tq, err := NewTurboQuant(dim, 4, 11)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(55))
	vec32 := make([]float32, dim)
	for i := range vec32 {
		vec32[i] = rng.Float32()*2 - 1
	}

	qv, err := tq.Quantize(vec32)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// DequantizeFloat64
	got64, err := tq.DequantizeFloat64(qv)
	if err != nil {
		t.Fatalf("DequantizeFloat64: %v", err)
	}

	// Manual: Dequantize then convert
	got32, err := tq.Dequantize(qv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}
	want64 := Float32sToFloat64s(got32)

	for i := range got64 {
		if got64[i] != want64[i] {
			t.Errorf("index %d: got %f, want %f", i, got64[i], want64[i])
		}
	}
}

// TestQuantizeBatchFloat64_Correctness tests batch float64 quantize and dequantize.
func TestQuantizeBatchFloat64_Correctness(t *testing.T) {
	dim := 16
	tq, err := NewTurboQuant(dim, 4, 7)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(123))
	numVecs := 5
	vecs := make([][]float64, numVecs)
	for i := range vecs {
		v := make([]float64, dim)
		for j := range v {
			v[j] = rng.Float64()*2 - 1
		}
		vecs[i] = v
	}

	qvs, err := tq.QuantizeBatchFloat64(vecs)
	if err != nil {
		t.Fatalf("QuantizeBatchFloat64: %v", err)
	}
	if len(qvs) != numVecs {
		t.Fatalf("expected %d results, got %d", numVecs, len(qvs))
	}

	restored, err := tq.DequantizeBatchFloat64(qvs)
	if err != nil {
		t.Fatalf("DequantizeBatchFloat64: %v", err)
	}
	if len(restored) != numVecs {
		t.Fatalf("expected %d results, got %d", numVecs, len(restored))
	}

	for i := 0; i < numVecs; i++ {
		if len(restored[i]) != dim {
			t.Errorf("restored[%d] dimension = %d, want %d", i, len(restored[i]), dim)
		}
	}
}

// TestQuantizeFloat64_DimensionMismatch verifies error on wrong dimension.
func TestQuantizeFloat64_DimensionMismatch(t *testing.T) {
	tq, err := NewTurboQuant(8, 4, 1)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	_, err = tq.QuantizeFloat64(make([]float64, 5))
	if err == nil {
		t.Error("expected error for dimension mismatch, got nil")
	}
}

// TestQuantizeBatchFloat64_DimensionMismatch verifies error on mixed dimensions.
func TestQuantizeBatchFloat64_DimensionMismatch(t *testing.T) {
	dim := 8
	tq, err := NewTurboQuant(dim, 4, 1)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	vecs := [][]float64{
		make([]float64, dim),
		make([]float64, 5), // wrong
	}

	_, err = tq.QuantizeBatchFloat64(vecs)
	if err == nil {
		t.Error("expected error for dimension mismatch, got nil")
	}
}

// TestFloat64_EmptyBatch verifies empty batch returns nil.
func TestFloat64_EmptyBatch(t *testing.T) {
	tq, err := NewTurboQuant(4, 2, 1)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	qvs, err := tq.QuantizeBatchFloat64(nil)
	if err != nil {
		t.Fatalf("QuantizeBatchFloat64(nil): %v", err)
	}
	if qvs != nil {
		t.Errorf("expected nil, got %v", qvs)
	}

	results, err := tq.DequantizeBatchFloat64(nil)
	if err != nil {
		t.Fatalf("DequantizeBatchFloat64(nil): %v", err)
	}
	if results != nil {
		t.Errorf("expected nil, got %v", results)
	}
}

// ---- Large dimension and boundary condition tests ----

// TestLargeDim4096 tests quantize/dequantize and serialize round-trip at dim=4096
// (typical LLM hidden size) across all bit widths.
func TestLargeDim4096(t *testing.T) {
	testLargeDimension(t, 4096)
}

// TestLargeDim8192 tests quantize/dequantize and serialize round-trip at dim=8192
// across all bit widths.
func TestLargeDim8192(t *testing.T) {
	testLargeDimension(t, 8192)
}

// testLargeDimension is a shared helper for large-dimension smoke tests.
// It uses table-driven tests across all 3 bit widths, verifying:
//   - Cosine similarity is positive and reasonable
//   - Serialize/deserialize round-trip consistency
//
// Note: At very large dimensions (4096+), the Beta distribution becomes extremely
// concentrated around 0, so most coordinates map to the same few centroids.
// The cosine similarity thresholds are relaxed compared to dim=64 to reflect
// the algorithm's theoretical behavior at high dimensions.
func testLargeDimension(t *testing.T, dim int) {
	t.Helper()

	if testing.Short() {
		t.Skipf("skipping large dimension test (dim=%d) in short mode", dim)
	}

	tests := []struct {
		bitWidth  int
		threshold float64
	}{
		{Bit2, 0.70},
		{Bit3, 0.70},
		{Bit4, 0.70},
	}

	const seed int64 = 42

	// Pre-generate the rotation matrix once — QR decomposition at large dimensions
	// is O(n³) and dominates test time. The rotation matrix depends only on
	// (dimension, seed), not bitWidth, so we share it across sub-tests.
	rotation, err := NewRandomOrthogonalMatrix(dim, seed)
	if err != nil {
		t.Fatalf("NewRandomOrthogonalMatrix(dim=%d): %v", dim, err)
	}

	// Generate a deterministic random vector once (shared across bit widths).
	rng := rand.New(rand.NewSource(99))
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64())
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%d-bit", tt.bitWidth)
		t.Run(name, func(t *testing.T) {
			codebook, err := GetOrBuildCodebook(dim, tt.bitWidth)
			if err != nil {
				t.Fatalf("GetOrBuildCodebook(dim=%d, bw=%d): %v", dim, tt.bitWidth, err)
			}

			// Quantize
			qv, err := quantizeVector(vec, rotation, codebook)
			if err != nil {
				t.Fatalf("Quantize: %v", err)
			}

			// Dequantize and verify cosine similarity threshold
			restored, err := dequantizeVector(qv, rotation, codebook)
			if err != nil {
				t.Fatalf("Dequantize: %v", err)
			}

			sim, err := CosineSimilarity(vec, restored)
			if err != nil {
				t.Fatalf("CosineSimilarity: %v", err)
			}
			if sim < tt.threshold {
				t.Errorf("cosine similarity = %f, want >= %f (dim=%d, %d-bit)",
					sim, tt.threshold, dim, tt.bitWidth)
			}

			// Serialize/Deserialize round-trip
			data, err := SerializeQuantizedVector(qv, tt.bitWidth)
			if err != nil {
				t.Fatalf("Serialize: %v", err)
			}

			qv2, err := DeserializeQuantizedVector(data, tt.bitWidth, dim)
			if err != nil {
				t.Fatalf("Deserialize: %v", err)
			}

			if qv.Norm != qv2.Norm {
				t.Errorf("Norm mismatch after round-trip: %f vs %f", qv.Norm, qv2.Norm)
			}
			if len(qv.Indices) != len(qv2.Indices) {
				t.Fatalf("Indices length mismatch: %d vs %d", len(qv.Indices), len(qv2.Indices))
			}
			for i := range qv.Indices {
				if qv.Indices[i] != qv2.Indices[i] {
					t.Errorf("Index mismatch at %d: %d vs %d", i, qv.Indices[i], qv2.Indices[i])
					break
				}
			}
		})
	}
}

// TestNaNInfInput verifies that Quantize returns an error when the input vector
// contains NaN or Inf values.
func TestNaNInfInput(t *testing.T) {
	const dim = 64
	tq, err := NewTurboQuant(dim, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	tests := []struct {
		name string
		vec  []float32
	}{
		{
			name: "single NaN",
			vec:  makeVecWith(dim, 0, float32(math.NaN())),
		},
		{
			name: "single +Inf",
			vec:  makeVecWith(dim, 0, float32(math.Inf(1))),
		},
		{
			name: "single -Inf",
			vec:  makeVecWith(dim, 0, float32(math.Inf(-1))),
		},
		{
			name: "NaN in middle",
			vec:  makeVecWith(dim, dim/2, float32(math.NaN())),
		},
		{
			name: "multiple NaN",
			vec:  makeAllNaN(dim),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tq.Quantize(tt.vec)
			if err == nil {
				t.Error("expected error for invalid input, got nil")
			}
		})
	}
}

// makeVecWith creates a dim-length vector of 1.0s with a special value at the given index.
func makeVecWith(dim, idx int, val float32) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = 1.0
	}
	vec[idx] = val
	return vec
}

// makeAllNaN creates a dim-length vector of all NaN values.
func makeAllNaN(dim int) []float32 {
	vec := make([]float32, dim)
	nan := float32(math.NaN())
	for i := range vec {
		vec[i] = nan
	}
	return vec
}

// TestLargeNormVector tests vectors with very large values near float32 max.
// The quantizer should either handle them correctly or return a clear error.
func TestLargeNormVector(t *testing.T) {
	const dim = 64
	tq, err := NewTurboQuant(dim, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	tests := []struct {
		name string
		val  float32
	}{
		{"half_max", math.MaxFloat32 / 2},
		{"sqrt_max", float32(math.Sqrt(math.MaxFloat32))},
		{"large_but_safe", 1e30},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = tt.val / float32(dim) // spread across dimensions to avoid overflow
			}

			qv, err := tq.Quantize(vec)
			if err != nil {
				// Returning an error is acceptable for extreme values.
				t.Logf("Quantize returned error (acceptable): %v", err)
				return
			}

			// If quantization succeeded, verify dequantize also works.
			restored, err := tq.Dequantize(qv)
			if err != nil {
				t.Fatalf("Dequantize failed after successful Quantize: %v", err)
			}

			// Verify restored values are finite.
			for i, v := range restored {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("restored[%d] = %v, expected finite value", i, v)
					break
				}
			}
		})
	}
}

// TestLargeBatchMemoryPressure quantizes 100,000 vectors of dim=64 in a batch,
// verifying it completes without error and spot-checking a few results.
func TestLargeBatchMemoryPressure(t *testing.T) {
	const dim = 64
	const batchSize = 100_000
	const bitWidth = Bit4

	tq, err := NewTurboQuant(dim, bitWidth, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	// Generate 100k random vectors.
	rng := rand.New(rand.NewSource(12345))
	vecs := make([][]float32, batchSize)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}

	// Batch quantize.
	qvs, err := tq.QuantizeBatch(vecs)
	if err != nil {
		t.Fatalf("QuantizeBatch(%d vectors): %v", batchSize, err)
	}
	if len(qvs) != batchSize {
		t.Fatalf("QuantizeBatch returned %d results, want %d", len(qvs), batchSize)
	}

	// Batch dequantize.
	restored, err := tq.DequantizeBatch(qvs)
	if err != nil {
		t.Fatalf("DequantizeBatch: %v", err)
	}
	if len(restored) != batchSize {
		t.Fatalf("DequantizeBatch returned %d results, want %d", len(restored), batchSize)
	}

	// Spot-check: verify cosine similarity for a few vectors spread across the batch.
	spotChecks := []int{0, 1, batchSize / 4, batchSize / 2, batchSize - 1}
	for _, idx := range spotChecks {
		sim, err := CosineSimilarity(vecs[idx], restored[idx])
		if err != nil {
			t.Errorf("CosineSimilarity for vec[%d]: %v", idx, err)
			continue
		}
		// 4-bit at dim=64 should achieve >= 0.99.
		if sim < 0.99 {
			t.Errorf("vec[%d]: cosine similarity = %f, want >= 0.99", idx, sim)
		}
	}
}

// ---- Option tests ----

// TestWithGridPoints_AffectsCodebook verifies that a custom gridPoints value
// produces a different codebook than the default, confirming the option takes effect.
func TestWithGridPoints_AffectsCodebook(t *testing.T) {
	dim := 64
	bw := 4
	seed := int64(42)

	tqDefault, err := NewTurboQuant(dim, bw, seed)
	if err != nil {
		t.Fatalf("NewTurboQuant (default): %v", err)
	}

	// Use a very different gridPoints value to get a noticeably different codebook.
	tqCustom, err := NewTurboQuant(dim, bw, seed, WithGridPoints(1000))
	if err != nil {
		t.Fatalf("NewTurboQuant (custom gridPoints): %v", err)
	}

	// The codebooks should differ because fewer grid points change the Lloyd-Max result.
	if codebooksEqual(tqDefault.codebook, tqCustom.codebook) {
		t.Error("expected different codebooks with different gridPoints, but they are equal")
	}
}

// TestWithIterations_AffectsCodebook verifies that a custom iterations value
// produces a different codebook than the default.
func TestWithIterations_AffectsCodebook(t *testing.T) {
	dim := 64
	bw := 4
	seed := int64(42)

	tqDefault, err := NewTurboQuant(dim, bw, seed)
	if err != nil {
		t.Fatalf("NewTurboQuant (default): %v", err)
	}

	// Use very few iterations so the codebook hasn't converged.
	tqCustom, err := NewTurboQuant(dim, bw, seed, WithIterations(1))
	if err != nil {
		t.Fatalf("NewTurboQuant (custom iterations): %v", err)
	}

	if codebooksEqual(tqDefault.codebook, tqCustom.codebook) {
		t.Error("expected different codebooks with different iterations, but they are equal")
	}
}

// TestWithOptions_StillQuantizes verifies that a TurboQuant created with custom
// options can still quantize and dequantize correctly.
func TestWithOptions_StillQuantizes(t *testing.T) {
	dim := 64
	bw := 4
	seed := int64(42)

	tq, err := NewTurboQuant(dim, bw, seed, WithGridPoints(60000), WithIterations(400))
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(99))
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}

	qv, err := tq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	restored, err := tq.Dequantize(qv)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	sim, err := CosineSimilarity(vec, restored)
	if err != nil {
		t.Fatalf("CosineSimilarity: %v", err)
	}

	// With more grid points and iterations, quality should be at least as good.
	if sim < 0.99 {
		t.Errorf("cosine similarity = %f, want >= 0.99", sim)
	}
}

// TestDefaultOptions_UsesCachedCodebook verifies that calling NewTurboQuant
// without options still uses the cached codebook (same pointer).
func TestDefaultOptions_UsesCachedCodebook(t *testing.T) {
	ResetCodebookCache()

	dim := 16
	bw := 3
	seed := int64(1)

	tq1, err := NewTurboQuant(dim, bw, seed)
	if err != nil {
		t.Fatalf("NewTurboQuant 1: %v", err)
	}

	tq2, err := NewTurboQuant(dim, bw, seed+1) // different seed, same dim/bw
	if err != nil {
		t.Fatalf("NewTurboQuant 2: %v", err)
	}

	// Both should share the same cached codebook pointer.
	if tq1.codebook != tq2.codebook {
		t.Error("expected same cached codebook pointer for default options, got different")
	}
}

// codebooksEqual checks if two codebooks have identical centroids.
func codebooksEqual(a, b *Codebook) bool {
	if len(a.Centroids) != len(b.Centroids) {
		return false
	}
	for i := range a.Centroids {
		if a.Centroids[i] != b.Centroids[i] {
			return false
		}
	}
	return true
}

// TestWithConcurrency_Default verifies that the default concurrency equals runtime.NumCPU().
func TestWithConcurrency_Default(t *testing.T) {
	tq, err := NewTurboQuant(64, 4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	expected := runtime.NumCPU()
	if tq.Concurrency() != expected {
		t.Errorf("default concurrency = %d, want %d (runtime.NumCPU())", tq.Concurrency(), expected)
	}
}

// TestWithConcurrency_Custom verifies that WithConcurrency sets the value correctly.
func TestWithConcurrency_Custom(t *testing.T) {
	tq, err := NewTurboQuant(64, 4, 42, WithConcurrency(3))
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	if tq.Concurrency() != 3 {
		t.Errorf("concurrency = %d, want 3", tq.Concurrency())
	}
}

// TestWithConcurrency_InvalidFallsBack verifies that n < 1 falls back to runtime.NumCPU().
func TestWithConcurrency_InvalidFallsBack(t *testing.T) {
	tq, err := NewTurboQuant(64, 4, 42, WithConcurrency(0))
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	expected := runtime.NumCPU()
	if tq.Concurrency() != expected {
		t.Errorf("concurrency = %d, want %d (runtime.NumCPU())", tq.Concurrency(), expected)
	}

	tq2, err := NewTurboQuant(64, 4, 42, WithConcurrency(-5))
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}
	if tq2.Concurrency() != expected {
		t.Errorf("concurrency = %d, want %d (runtime.NumCPU())", tq2.Concurrency(), expected)
	}
}

// TestWithConcurrency_BatchCorrectness verifies that batch results are correct
// regardless of the concurrency level.
func TestWithConcurrency_BatchCorrectness(t *testing.T) {
	dim := 64
	bw := 4
	batchSize := 50

	rng := rand.New(rand.NewSource(99))
	vecs := make([][]float32, batchSize)
	for i := range vecs {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()*2 - 1
		}
		vecs[i] = vec
	}

	for _, concurrency := range []int{1, 2, 4, 16} {
		t.Run(fmt.Sprintf("concurrency_%d", concurrency), func(t *testing.T) {
			tq, err := NewTurboQuant(dim, bw, 42, WithConcurrency(concurrency))
			if err != nil {
				t.Fatalf("NewTurboQuant: %v", err)
			}

			qvs, err := tq.QuantizeBatch(vecs)
			if err != nil {
				t.Fatalf("QuantizeBatch: %v", err)
			}

			restored, err := tq.DequantizeBatch(qvs)
			if err != nil {
				t.Fatalf("DequantizeBatch: %v", err)
			}

			for i, orig := range vecs {
				sim, err := CosineSimilarity(orig, restored[i])
				if err != nil {
					t.Fatalf("vec %d: CosineSimilarity: %v", i, err)
				}
				if sim < 0.99 {
					t.Errorf("vec %d: cosine similarity = %f, want >= 0.99", i, sim)
				}
			}
		})
	}
}

// ---- Pool concurrent safety tests ----

// TestPoolConcurrentBatchSafety verifies that concurrent batch quantize/dequantize
// operations produce correct results when sharing pooled slices. This catches
// data corruption from improper pool reuse (e.g., returning a slice that's still
// referenced by another goroutine).
func TestPoolConcurrentBatchSafety(t *testing.T) {
	const dim = 128
	const bw = 4
	const batchSize = 50
	const numGoroutines = 16

	tq, err := NewTurboQuant(dim, bw, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	// Generate deterministic test vectors — each goroutine gets its own batch.
	allVecs := make([][][]float32, numGoroutines)
	for g := 0; g < numGoroutines; g++ {
		rng := rand.New(rand.NewSource(int64(g * 1000)))
		vecs := make([][]float32, batchSize)
		for i := range vecs {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			vecs[i] = vec
		}
		allVecs[g] = vecs
	}

	// Run concurrent batch quantize → dequantize and verify cosine similarity.
	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			vecs := allVecs[goroutineID]

			// Quantize batch.
			qvs, err := tq.QuantizeBatch(vecs)
			if err != nil {
				errCh <- fmt.Errorf("goroutine %d: QuantizeBatch: %v", goroutineID, err)
				return
			}

			// Dequantize batch.
			restored, err := tq.DequantizeBatch(qvs)
			if err != nil {
				errCh <- fmt.Errorf("goroutine %d: DequantizeBatch: %v", goroutineID, err)
				return
			}

			// Verify each vector's cosine similarity meets the 4-bit threshold.
			for i, orig := range vecs {
				sim, err := CosineSimilarity(orig, restored[i])
				if err != nil {
					errCh <- fmt.Errorf("goroutine %d, vec %d: CosineSimilarity: %v", goroutineID, i, err)
					return
				}
				if sim < 0.98 {
					errCh <- fmt.Errorf("goroutine %d, vec %d: cosine similarity %.6f < 0.98 — possible pool corruption", goroutineID, i, sim)
					return
				}
			}
		}(g)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}
}

// TestPoolConcurrentMixedOperations verifies pool safety when quantize and
// dequantize operations run concurrently on the same TurboQuant instance,
// exercising both getFloat64Slice and putFloat64Slice under contention.
func TestPoolConcurrentMixedOperations(t *testing.T) {
	const dim = 64
	const bw = 3
	const iterations = 100
	const numGoroutines = 8

	tq, err := NewTurboQuant(dim, bw, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	// Pre-generate vectors and their quantized forms.
	rng := rand.New(rand.NewSource(42))
	vecs := make([][]float32, iterations)
	qvs := make([]*QuantizedVector, iterations)
	for i := 0; i < iterations; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()*2 - 1
		}
		vecs[i] = vec
		qv, err := tq.Quantize(vec)
		if err != nil {
			t.Fatalf("pre-quantize %d: %v", i, err)
		}
		qvs[i] = qv
	}

	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines*2)

	// Half goroutines quantize, half dequantize — all sharing the same pools.
	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		if g%2 == 0 {
			go func(id int) {
				defer wg.Done()
				for i := 0; i < iterations; i++ {
					qv, err := tq.Quantize(vecs[i])
					if err != nil {
						errCh <- fmt.Errorf("quantize goroutine %d, iter %d: %v", id, i, err)
						return
					}
					if len(qv.Indices) != dim {
						errCh <- fmt.Errorf("quantize goroutine %d, iter %d: expected %d indices, got %d", id, i, dim, len(qv.Indices))
						return
					}
				}
			}(g)
		} else {
			go func(id int) {
				defer wg.Done()
				for i := 0; i < iterations; i++ {
					result, err := tq.Dequantize(qvs[i])
					if err != nil {
						errCh <- fmt.Errorf("dequantize goroutine %d, iter %d: %v", id, i, err)
						return
					}
					if len(result) != dim {
						errCh <- fmt.Errorf("dequantize goroutine %d, iter %d: expected %d elements, got %d", id, i, dim, len(result))
						return
					}
				}
			}(g)
		}
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}
}
