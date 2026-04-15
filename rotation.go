package turboquant

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Matrix represents a dense matrix, internally using gonum/mat.Dense.
type Matrix struct {
	data *mat.Dense
	dim  int
}

// Apply multiplies the matrix by a vector: result = M * vec.
func (m *Matrix) Apply(vec []float64) []float64 {
	out := make([]float64, m.dim)
	m.ApplyInto(vec, out)
	return out
}

// ApplyInto multiplies the matrix by a vector, writing the result into dst.
// dst must have length >= m.dim.
func (m *Matrix) ApplyInto(vec, dst []float64) {
	v := mat.NewVecDense(len(vec), vec)
	result := mat.NewVecDense(m.dim, dst[:m.dim])
	result.MulVec(m.data, v)
}

// ApplyTranspose multiplies the matrix transpose by a vector: result = M^T * vec.
func (m *Matrix) ApplyTranspose(vec []float64) []float64 {
	out := make([]float64, m.dim)
	m.ApplyTransposeInto(vec, out)
	return out
}

// ApplyTransposeInto multiplies the matrix transpose by a vector, writing the result into dst.
// dst must have length >= m.dim.
func (m *Matrix) ApplyTransposeInto(vec, dst []float64) {
	v := mat.NewVecDense(len(vec), vec)
	result := mat.NewVecDense(m.dim, dst[:m.dim])
	result.MulVec(m.data.T(), v)
}

// NewRandomOrthogonalMatrix generates a random orthogonal matrix.
// Obtained by QR decomposition of a random Gaussian matrix.
// Same seed produces the same matrix. Returns an error if dimension < 2.
func NewRandomOrthogonalMatrix(dimension int, seed int64) (*Matrix, error) {
	if err := ValidateDimension(dimension); err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewSource(seed))

	// Generate dimension x dimension random Gaussian matrix
	data := make([]float64, dimension*dimension)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	gaussian := mat.NewDense(dimension, dimension, data)

	// QR decomposition
	var qr mat.QR
	qr.Factorize(gaussian)

	// Extract Q matrix
	var q mat.Dense
	qr.QTo(&q)

	return &Matrix{data: &q, dim: dimension}, nil
}
