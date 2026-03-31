package turboquant

import (
	"math/rand"
	"testing"
	"testing/quick"
)

// TestProperty_SerializeRoundTrip verifies that for any valid QuantizedVector,
// serializing then deserializing produces a result exactly equal to the original.
//
// **Validates: Requirements 9.3**
func TestProperty_SerializeRoundTrip(t *testing.T) {
	bitWidths := []int{Bit2, Bit3, Bit4}

	prop := func(seed uint32) bool {
		rng := rand.New(rand.NewSource(int64(seed)))

		// Derive dimension in [2, 256] and pick a random bitWidth from {2, 3, 4}.
		dimension := rng.Intn(255) + 2 // [2, 256]
		bitWidth := bitWidths[rng.Intn(3)]
		maxIndex := uint8((1 << uint(bitWidth)) - 1)

		// Generate a random QuantizedVector.
		norm := rng.Float32() * 100.0 // arbitrary positive norm
		indices := make([]uint8, dimension)
		for i := range indices {
			indices[i] = uint8(rng.Intn(int(maxIndex) + 1))
		}

		original := &QuantizedVector{
			Norm:    norm,
			Indices: indices,
		}

		// Serialize.
		data, err := SerializeQuantizedVector(original, bitWidth)
		if err != nil {
			t.Logf("SerializeQuantizedVector error: %v (seed=%d, dim=%d, bw=%d)", err, seed, dimension, bitWidth)
			return false
		}

		// Deserialize.
		restored, err := DeserializeQuantizedVector(data, bitWidth, dimension)
		if err != nil {
			t.Logf("DeserializeQuantizedVector error: %v (seed=%d, dim=%d, bw=%d)", err, seed, dimension, bitWidth)
			return false
		}

		// Compare norm: must be exactly equal.
		if original.Norm != restored.Norm {
			t.Logf("norm mismatch: original=%v, restored=%v (seed=%d)", original.Norm, restored.Norm, seed)
			return false
		}

		// Compare indices: must be exactly equal element-by-element.
		if len(original.Indices) != len(restored.Indices) {
			t.Logf("indices length mismatch: original=%d, restored=%d (seed=%d)", len(original.Indices), len(restored.Indices), seed)
			return false
		}
		for i := range original.Indices {
			if original.Indices[i] != restored.Indices[i] {
				t.Logf("index mismatch at [%d]: original=%d, restored=%d (seed=%d, dim=%d, bw=%d)",
					i, original.Indices[i], restored.Indices[i], seed, dimension, bitWidth)
				return false
			}
		}

		return true
	}

	cfg := &quick.Config{MaxCount: 5}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Property failed (serialize round-trip consistency): %v", err)
	}
}

// TestDeserialize_WrongDataLength verifies that DeserializeQuantizedVector returns
// an error when the data length does not match the expected size.
func TestDeserialize_WrongDataLength(t *testing.T) {
	// For dimension=8, bitWidth=4: expected = 4 (norm) + 4 (8 indices / 2) = 8 bytes.
	// Provide only 5 bytes, which is wrong.
	data := []byte{0x00, 0x00, 0x80, 0x3F, 0xFF}
	_, err := DeserializeQuantizedVector(data, Bit4, 8)
	if err == nil {
		t.Error("expected error for wrong data length, got nil")
	}
}
