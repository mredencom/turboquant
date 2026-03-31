package turboquant

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// FuzzSerializeDeserialize verifies that serializing then deserializing a
// QuantizedVector always produces a result identical to the original.
// Uses Go native fuzz testing (Go 1.18+).
func FuzzSerializeDeserialize(f *testing.F) {
	// Seed corpus: a few representative cases.
	// Each seed is a uint32 used to derive random QuantizedVector data.
	f.Add(uint32(0))
	f.Add(uint32(42))
	f.Add(uint32(12345))
	f.Add(uint32(math.MaxUint32))

	f.Fuzz(func(t *testing.T, seed uint32) {
		rng := rand.New(rand.NewSource(int64(seed)))

		bitWidths := []int{Bit2, Bit3, Bit4}
		bitWidth := bitWidths[rng.Intn(3)]
		maxIndex := uint8((1 << uint(bitWidth)) - 1)

		// Dimension in [2, 256].
		dimension := rng.Intn(255) + 2

		// Random norm (including 0 and negative).
		norm := rng.Float32()*200.0 - 100.0

		indices := make([]uint8, dimension)
		for i := range indices {
			indices[i] = uint8(rng.Intn(int(maxIndex) + 1))
		}

		original := &QuantizedVector{Norm: norm, Indices: indices}

		data, err := SerializeQuantizedVector(original, bitWidth)
		if err != nil {
			t.Fatalf("serialize error: %v", err)
		}

		restored, err := DeserializeQuantizedVector(data, bitWidth, dimension)
		if err != nil {
			t.Fatalf("deserialize error: %v", err)
		}

		if original.Norm != restored.Norm {
			t.Fatalf("norm mismatch: got %v, want %v", restored.Norm, original.Norm)
		}
		if len(original.Indices) != len(restored.Indices) {
			t.Fatalf("indices length mismatch: got %d, want %d", len(restored.Indices), len(original.Indices))
		}
		for i := range original.Indices {
			if original.Indices[i] != restored.Indices[i] {
				t.Fatalf("index[%d] mismatch: got %d, want %d (dim=%d, bw=%d)",
					i, restored.Indices[i], original.Indices[i], dimension, bitWidth)
			}
		}
	})
}

// FuzzQuantizeDequantize verifies that quantizing then dequantizing a random
// float32 vector produces no errors and yields a result with positive cosine
// similarity to the original (for non-zero vectors).
func FuzzQuantizeDequantize(f *testing.F) {
	// Seed corpus: uint64 seeds used to generate random float32 vectors.
	f.Add(uint64(0))
	f.Add(uint64(42))
	f.Add(uint64(99999))

	// Pre-create a TurboQuant instance with dimension=64, 4-bit, seed=1.
	const dim = 64
	tq, err := NewTurboQuant(dim, Bit4, 1)
	if err != nil {
		f.Fatalf("failed to create TurboQuant: %v", err)
	}

	f.Fuzz(func(t *testing.T, seed uint64) {
		rng := rand.New(rand.NewSource(int64(seed)))

		// Generate a random float32 vector of the fixed dimension.
		vec := make([]float32, dim)
		allZero := true
		for i := range vec {
			vec[i] = rng.Float32()*2.0 - 1.0 // values in [-1, 1]
			if vec[i] != 0 {
				allZero = false
			}
		}

		qv, err := tq.Quantize(vec)
		if err != nil {
			t.Fatalf("quantize error: %v", err)
		}

		restored, err := tq.Dequantize(qv)
		if err != nil {
			t.Fatalf("dequantize error: %v", err)
		}

		if len(restored) != dim {
			t.Fatalf("restored dimension mismatch: got %d, want %d", len(restored), dim)
		}

		// For non-zero vectors, cosine similarity should be > 0.
		if !allZero {
			sim, err := CosineSimilarity(vec, restored)
			if err != nil {
				t.Fatalf("cosine similarity error: %v", err)
			}
			if sim <= 0 {
				t.Fatalf("cosine similarity should be > 0 for non-zero vector, got %f", sim)
			}
		}
	})
}

// FuzzDeserialize feeds random bytes to DeserializeQuantizedVector and verifies
// that it either returns a valid result or an error — it must never panic.
func FuzzDeserialize(f *testing.F) {
	// Seed corpus: various byte slices including empty, short, and structured data.
	f.Add([]byte{})
	f.Add([]byte{0x00})
	f.Add([]byte{0x00, 0x00, 0x80, 0x3F}) // float32 1.0 in little-endian
	f.Add(make([]byte, 100))

	// Add a valid serialized vector as seed.
	norm := float32(1.5)
	var normBytes [4]byte
	binary.LittleEndian.PutUint32(normBytes[:], math.Float32bits(norm))
	validData := append(normBytes[:], 0x12, 0x34, 0x56, 0x78)
	f.Add(validData)

	f.Fuzz(func(t *testing.T, data []byte) {
		// Try all three bit widths with a fixed dimension.
		// The function should never panic regardless of input.
		for _, bw := range []int{Bit2, Bit3, Bit4} {
			for _, dim := range []int{2, 8, 64} {
				qv, err := DeserializeQuantizedVector(data, bw, dim)
				// Either err != nil (invalid data) or qv is a valid result.
				if err == nil && qv == nil {
					t.Fatalf("got nil result with nil error for bw=%d dim=%d", bw, dim)
				}
			}
		}
	})
}
