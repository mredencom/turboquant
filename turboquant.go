package turboquant

import (
	"fmt"
	"sync"
)

// TurboQuant is the core entry point of the SDK, encapsulating all quantization functionality.
type TurboQuant struct {
	dimension int
	bitWidth  int
	codebook  *Codebook
	rotation  *Matrix
}

// NewTurboQuant creates and initializes a quantizer instance.
// dimension: vector dimension, must be >= 2
// bitWidth: quantization bit width, must be 2, 3, or 4
// seed: random seed for rotation matrix generation; same seed produces same matrix
func NewTurboQuant(dimension, bitWidth int, seed int64) (*TurboQuant, error) {
	if err := ValidateDimension(dimension); err != nil {
		return nil, fmt.Errorf("NewTurboQuant: %w", err)
	}
	if err := ValidateBitWidth(bitWidth); err != nil {
		return nil, fmt.Errorf("NewTurboQuant: %w", err)
	}

	codebook, err := GetOrBuildCodebook(dimension, bitWidth)
	if err != nil {
		return nil, fmt.Errorf("NewTurboQuant: failed to build codebook: %w", err)
	}

	rotation, err := NewRandomOrthogonalMatrix(dimension, seed)
	if err != nil {
		return nil, fmt.Errorf("NewTurboQuant: failed to generate rotation matrix: %w", err)
	}

	return &TurboQuant{
		dimension: dimension,
		bitWidth:  bitWidth,
		codebook:  codebook,
		rotation:  rotation,
	}, nil
}

// Quantize quantizes a single float32 vector into a QuantizedVector.
func (tq *TurboQuant) Quantize(vec []float32) (*QuantizedVector, error) {
	return quantizeVector(vec, tq.rotation, tq.codebook)
}

// Dequantize reconstructs a float32 vector from a QuantizedVector.
func (tq *TurboQuant) Dequantize(qv *QuantizedVector) ([]float32, error) {
	return dequantizeVector(qv, tq.rotation, tq.codebook)
}

// Serialize serializes a QuantizedVector into a compact binary byte slice.
func (tq *TurboQuant) Serialize(qv *QuantizedVector) ([]byte, error) {
	return SerializeQuantizedVector(qv, tq.bitWidth)
}

// Deserialize deserializes a binary byte slice into a QuantizedVector.
func (tq *TurboQuant) Deserialize(data []byte) (*QuantizedVector, error) {
	return DeserializeQuantizedVector(data, tq.bitWidth, tq.dimension)
}

// CompressionRatio returns the theoretical compression ratio for the current configuration.
func (tq *TurboQuant) CompressionRatio() float64 {
	return CompressionRatio(tq.dimension, tq.bitWidth)
}

// Dimension returns the vector dimension of this quantizer.
func (tq *TurboQuant) Dimension() int {
	return tq.dimension
}

// BitWidth returns the quantization bit width of this quantizer.
func (tq *TurboQuant) BitWidth() int {
	return tq.bitWidth
}

// QuantizeBatch quantizes multiple vectors concurrently using goroutines.
// All vectors must have the same dimension as the TurboQuant instance.
// If any vector has a mismatched dimension, returns an error indicating the first such index.
func (tq *TurboQuant) QuantizeBatch(vecs [][]float32) ([]*QuantizedVector, error) {
	if len(vecs) == 0 {
		return nil, nil
	}

	// Check all vector dimensions first, returning the first mismatch index.
	for i, vec := range vecs {
		if len(vec) != tq.dimension {
			return nil, fmt.Errorf("QuantizeBatch: vector at index %d has dimension %d, expected %d", i, len(vec), tq.dimension)
		}
	}

	results := make([]*QuantizedVector, len(vecs))
	errs := make([]error, len(vecs))

	var wg sync.WaitGroup
	wg.Add(len(vecs))

	for i, vec := range vecs {
		go func(idx int, v []float32) {
			defer wg.Done()
			qv, err := quantizeVector(v, tq.rotation, tq.codebook)
			if err != nil {
				errs[idx] = err
				return
			}
			results[idx] = qv
		}(i, vec)
	}

	wg.Wait()

	// Return the first error encountered.
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// DequantizeBatch dequantizes multiple QuantizedVectors concurrently using goroutines.
func (tq *TurboQuant) DequantizeBatch(qvs []*QuantizedVector) ([][]float32, error) {
	if len(qvs) == 0 {
		return nil, nil
	}

	results := make([][]float32, len(qvs))
	errs := make([]error, len(qvs))

	var wg sync.WaitGroup
	wg.Add(len(qvs))

	for i, qv := range qvs {
		go func(idx int, q *QuantizedVector) {
			defer wg.Done()
			vec, err := dequantizeVector(q, tq.rotation, tq.codebook)
			if err != nil {
				errs[idx] = err
				return
			}
			results[idx] = vec
		}(i, qv)
	}

	wg.Wait()

	// Return the first error encountered.
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}
