package turboquant

import (
	"math/rand"
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

// --- Integration Tests (Task 9.4) ---

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
