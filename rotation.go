package turboquant

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Matrix 表示一个 dense 矩阵，内部使用 gonum/mat.Dense。
type Matrix struct {
	data *mat.Dense
	dim  int
}

// Apply 将矩阵应用于向量：result = M · vec。
func (m *Matrix) Apply(vec []float64) []float64 {
	v := mat.NewVecDense(len(vec), vec)
	result := mat.NewVecDense(m.dim, nil)
	result.MulVec(m.data, v)
	out := make([]float64, m.dim)
	for i := range out {
		out[i] = result.AtVec(i)
	}
	return out
}

// ApplyTranspose 将矩阵转置应用于向量：result = M^T · vec。
func (m *Matrix) ApplyTranspose(vec []float64) []float64 {
	v := mat.NewVecDense(len(vec), vec)
	result := mat.NewVecDense(m.dim, nil)
	result.MulVec(m.data.T(), v)
	out := make([]float64, m.dim)
	for i := range out {
		out[i] = result.AtVec(i)
	}
	return out
}

// NewRandomOrthogonalMatrix 生成随机正交矩阵。
// 通过对随机高斯矩阵做 QR 分解获得 Q 矩阵。
// 相同 seed 产生相同矩阵，dimension < 2 时返回参数错误。
func NewRandomOrthogonalMatrix(dimension int, seed int64) (*Matrix, error) {
	if err := ValidateDimension(dimension); err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewSource(seed))

	// 生成 dimension × dimension 随机高斯矩阵
	data := make([]float64, dimension*dimension)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	gaussian := mat.NewDense(dimension, dimension, data)

	// QR 分解
	var qr mat.QR
	qr.Factorize(gaussian)

	// 提取 Q 矩阵
	var q mat.Dense
	qr.QTo(&q)

	return &Matrix{data: &q, dim: dimension}, nil
}
