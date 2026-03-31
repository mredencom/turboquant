package turboquant

import (
	"math"
	"testing"
)

func TestBetaPDF_KnownValues(t *testing.T) {
	// Beta(1,1) is Uniform(0,1), PDF = 1.0 everywhere in (0,1)
	got := BetaPDF(0.5, 1.0, 1.0)
	if math.Abs(got-1.0) > 1e-10 {
		t.Errorf("BetaPDF(0.5, 1, 1) = %v, want 1.0", got)
	}

	// Beta(2,2) at x=0.5: PDF = 6*0.5^1*0.5^1 = 1.5
	got = BetaPDF(0.5, 2.0, 2.0)
	if math.Abs(got-1.5) > 1e-10 {
		t.Errorf("BetaPDF(0.5, 2, 2) = %v, want 1.5", got)
	}

	// Beta(2,5) at x=0.3: known value ≈ 2.2689...
	// B(2,5) = Γ(2)Γ(5)/Γ(7) = 1*24/720 = 1/30
	// PDF = 0.3^1 * 0.7^4 / (1/30) = 0.3 * 0.2401 * 30 = 2.1609
	got = BetaPDF(0.3, 2.0, 5.0)
	expected := 0.3 * math.Pow(0.7, 4) * 30.0
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("BetaPDF(0.3, 2, 5) = %v, want %v", got, expected)
	}
}

func TestBetaPDF_Symmetry(t *testing.T) {
	// Beta(a,b) at x should equal Beta(b,a) at 1-x
	a, b, x := 3.0, 5.0, 0.3
	got1 := BetaPDF(x, a, b)
	got2 := BetaPDF(1-x, b, a)
	if math.Abs(got1-got2) > 1e-10 {
		t.Errorf("Symmetry failed: BetaPDF(%v,%v,%v)=%v != BetaPDF(%v,%v,%v)=%v", x, a, b, got1, 1-x, b, a, got2)
	}
}

func TestBetaPDF_EdgeCases(t *testing.T) {
	// x <= 0 returns 0
	if got := BetaPDF(0.0, 2.0, 2.0); got != 0.0 {
		t.Errorf("BetaPDF(0, 2, 2) = %v, want 0.0", got)
	}
	if got := BetaPDF(-0.5, 2.0, 2.0); got != 0.0 {
		t.Errorf("BetaPDF(-0.5, 2, 2) = %v, want 0.0", got)
	}

	// x >= 1 returns 0
	if got := BetaPDF(1.0, 2.0, 2.0); got != 0.0 {
		t.Errorf("BetaPDF(1, 2, 2) = %v, want 0.0", got)
	}
	if got := BetaPDF(1.5, 2.0, 2.0); got != 0.0 {
		t.Errorf("BetaPDF(1.5, 2, 2) = %v, want 0.0", got)
	}
}

func TestBetaPDF_LargeParameters(t *testing.T) {
	// Test with large alpha/beta to verify no overflow via Lgamma
	got := BetaPDF(0.5, 100.0, 100.0)
	if math.IsNaN(got) || math.IsInf(got, 0) || got <= 0 {
		t.Errorf("BetaPDF(0.5, 100, 100) = %v, expected finite positive value", got)
	}
}

func TestCosineSimilarity_NormalCase(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	got, err := CosineSimilarity(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// dot=32, normA=sqrt(14), normB=sqrt(77), expected=32/sqrt(14*77)=32/sqrt(1078)
	expected := 32.0 / math.Sqrt(14.0*77.0)
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("CosineSimilarity([1,2,3],[4,5,6]) = %v, want %v", got, expected)
	}
}

func TestCosineSimilarity_IdenticalVectors(t *testing.T) {
	a := []float32{3, 4, 0}
	got, err := CosineSimilarity(a, a)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if math.Abs(got-1.0) > 1e-10 {
		t.Errorf("CosineSimilarity(a, a) = %v, want 1.0", got)
	}
}

func TestCosineSimilarity_OppositeVectors(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{-1, -2, -3}
	got, err := CosineSimilarity(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if math.Abs(got-(-1.0)) > 1e-10 {
		t.Errorf("CosineSimilarity(a, -a) = %v, want -1.0", got)
	}
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	a := []float32{1, 2, 3}
	zero := []float32{0, 0, 0}

	// One zero vector
	got, err := CosineSimilarity(a, zero)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 0.0 {
		t.Errorf("CosineSimilarity(a, zero) = %v, want 0.0", got)
	}

	// Both zero vectors
	got, err = CosineSimilarity(zero, zero)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 0.0 {
		t.Errorf("CosineSimilarity(zero, zero) = %v, want 0.0", got)
	}
}

func TestCosineSimilarity_DimensionMismatch(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5}
	_, err := CosineSimilarity(a, b)
	if err == nil {
		t.Error("expected dimension mismatch error, got nil")
	}
}

func TestCompressionRatio_KnownValues(t *testing.T) {
	// dimension=128, bitWidth=2 → (128*32)/(32+128*2) = 4096/288 ≈ 14.222
	got := CompressionRatio(128, 2)
	expected := 4096.0 / 288.0
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("CompressionRatio(128, 2) = %v, want %v", got, expected)
	}

	// dimension=128, bitWidth=4 → (128*32)/(32+128*4) = 4096/544 ≈ 7.529
	got = CompressionRatio(128, 4)
	expected = 4096.0 / 544.0
	if math.Abs(got-expected) > 1e-10 {
		t.Errorf("CompressionRatio(128, 4) = %v, want %v", got, expected)
	}
}
