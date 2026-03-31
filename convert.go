package turboquant

import (
	"math"
)

// Float64sToFloat32s converts a []float64 slice to []float32.
func Float64sToFloat32s(src []float64) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = float32(v)
	}
	return dst
}

// Float32sToFloat64s converts a []float32 slice to []float64.
func Float32sToFloat64s(src []float32) []float64 {
	dst := make([]float64, len(src))
	for i, v := range src {
		dst[i] = float64(v)
	}
	return dst
}

// IntsToFloat32s converts a []int slice to []float32.
func IntsToFloat32s(src []int) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = float32(v)
	}
	return dst
}

// Float32sToInts converts a []float32 slice back to []int by rounding.
func Float32sToInts(src []float32) []int {
	dst := make([]int, len(src))
	for i, v := range src {
		dst[i] = int(math.Round(float64(v)))
	}
	return dst
}

// BytesToFloat32s converts a []byte slice to []float32.
// Each byte value (0-255) is stored as a float32.
func BytesToFloat32s(src []byte) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = float32(v)
	}
	return dst
}

// Float32sToBytes converts a []float32 slice back to []byte by rounding and clamping to [0, 255].
func Float32sToBytes(src []float32) []byte {
	dst := make([]byte, len(src))
	for i, v := range src {
		r := math.Round(float64(v))
		if r < 0 {
			r = 0
		} else if r > 255 {
			r = 255
		}
		dst[i] = byte(r)
	}
	return dst
}

// StringToFloat32s converts a string to []float32 by treating each byte as a float32 value.
// This is a raw byte-level conversion, not a semantic embedding.
func StringToFloat32s(s string) []float32 {
	return BytesToFloat32s([]byte(s))
}

// Float32sToString converts a []float32 slice back to a string by rounding each value to a byte.
func Float32sToString(src []float32) string {
	return string(Float32sToBytes(src))
}
