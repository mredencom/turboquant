package turboquant

import (
	"math"
	"math/rand"
	"testing"
	"testing/quick"

	"gonum.org/v1/gonum/mat"
)

func TestMatrixApply(t *testing.T) {
	// 2x2 identity matrix: Apply should return the same vector
	identity := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	m := &Matrix{data: identity, dim: 2}

	vec := []float64{3.0, 4.0}
	result := m.Apply(vec)

	if len(result) != 2 {
		t.Fatalf("expected length 2, got %d", len(result))
	}
	if math.Abs(result[0]-3.0) > 1e-10 || math.Abs(result[1]-4.0) > 1e-10 {
		t.Errorf("identity Apply: expected [3, 4], got %v", result)
	}
}

func TestMatrixApplyNonIdentity(t *testing.T) {
	// [[1, 2], [3, 4]] * [1, 1] = [3, 7]
	m := &Matrix{
		data: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
		dim:  2,
	}

	result := m.Apply([]float64{1, 1})

	if math.Abs(result[0]-3.0) > 1e-10 {
		t.Errorf("expected result[0]=3, got %f", result[0])
	}
	if math.Abs(result[1]-7.0) > 1e-10 {
		t.Errorf("expected result[1]=7, got %f", result[1])
	}
}

func TestMatrixApplyTranspose(t *testing.T) {
	// M = [[1, 2], [3, 4]], M^T = [[1, 3], [2, 4]]
	// M^T * [1, 1] = [4, 6]
	m := &Matrix{
		data: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
		dim:  2,
	}

	result := m.ApplyTranspose([]float64{1, 1})

	if math.Abs(result[0]-4.0) > 1e-10 {
		t.Errorf("expected result[0]=4, got %f", result[0])
	}
	if math.Abs(result[1]-6.0) > 1e-10 {
		t.Errorf("expected result[1]=6, got %f", result[1])
	}
}

func TestMatrixApplyTransposeIdentity(t *testing.T) {
	identity := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	m := &Matrix{data: identity, dim: 3}

	vec := []float64{5.0, 6.0, 7.0}
	result := m.ApplyTranspose(vec)

	for i, v := range vec {
		if math.Abs(result[i]-v) > 1e-10 {
			t.Errorf("identity ApplyTranspose: index %d expected %f, got %f", i, v, result[i])
		}
	}
}

// --- Property-Based Tests ---

// TestProperty_OrthogonalityVerification verifies that for any valid dimension and seed,
// the generated matrix R satisfies R^T · R ≈ I (identity), with error < 1e-6.
//
// **Validates: Requirements 3.2**
func TestProperty_OrthogonalityVerification(t *testing.T) {
	prop := func(seed uint32) bool {
		rng := rand.New(rand.NewSource(int64(seed)))
		// dimension in [2, 64]
		dimension := rng.Intn(63) + 2

		m, err := NewRandomOrthogonalMatrix(dimension, int64(seed))
		if err != nil {
			t.Logf("NewRandomOrthogonalMatrix(%d, %d) error: %v", dimension, seed, err)
			return false
		}

		// Compute R^T · R
		var rtR mat.Dense
		rtR.Mul(m.data.T(), m.data)

		// Verify R^T · R ≈ I
		for i := 0; i < dimension; i++ {
			for j := 0; j < dimension; j++ {
				val := rtR.At(i, j)
				if i == j {
					// Diagonal should be ≈ 1.0
					if math.Abs(val-1.0) > 1e-6 {
						t.Logf("dim=%d seed=%d: R^T·R[%d][%d] = %v, want ≈ 1.0", dimension, seed, i, j, val)
						return false
					}
				} else {
					// Off-diagonal should be ≈ 0.0
					if math.Abs(val) > 1e-6 {
						t.Logf("dim=%d seed=%d: R^T·R[%d][%d] = %v, want ≈ 0.0", dimension, seed, i, j, val)
						return false
					}
				}
			}
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 3}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Orthogonality property failed: %v", err)
	}
}

// TestProperty_Reproducibility verifies that calling NewRandomOrthogonalMatrix
// with the same dimension and seed produces identical matrices.
//
// **Validates: Requirements 3.3**
func TestProperty_Reproducibility(t *testing.T) {
	prop := func(seed uint32) bool {
		rng := rand.New(rand.NewSource(int64(seed)))
		// dimension in [2, 64]
		dimension := rng.Intn(63) + 2

		m1, err := NewRandomOrthogonalMatrix(dimension, int64(seed))
		if err != nil {
			t.Logf("first call error: %v", err)
			return false
		}

		m2, err := NewRandomOrthogonalMatrix(dimension, int64(seed))
		if err != nil {
			t.Logf("second call error: %v", err)
			return false
		}

		// Element-wise comparison
		for i := 0; i < dimension; i++ {
			for j := 0; j < dimension; j++ {
				v1 := m1.data.At(i, j)
				v2 := m2.data.At(i, j)
				if v1 != v2 {
					t.Logf("dim=%d seed=%d: m1[%d][%d]=%v != m2[%d][%d]=%v",
						dimension, seed, i, j, v1, i, j, v2)
					return false
				}
			}
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 3}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Reproducibility property failed: %v", err)
	}
}

// --- Edge Case Tests ---

func TestNewRandomOrthogonalMatrix_InvalidDimension(t *testing.T) {
	for _, dim := range []int{-1, 0, 1} {
		_, err := NewRandomOrthogonalMatrix(dim, 42)
		if err == nil {
			t.Errorf("NewRandomOrthogonalMatrix(%d, 42) expected error, got nil", dim)
		}
	}
}
