package turboquant

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
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

// ---- Streaming serialization tests ----

// TestSerializeTo_DeserializeFrom_RoundTrip verifies single vector round-trip
// via SerializeTo/DeserializeFrom using a bytes.Buffer.
func TestSerializeTo_DeserializeFrom_RoundTrip(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(123))
	vec := make([]float32, 64)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}

	qv, err := tq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Serialize to buffer.
	var buf bytes.Buffer
	if err := tq.SerializeTo(qv, &buf); err != nil {
		t.Fatalf("SerializeTo: %v", err)
	}

	// Deserialize from buffer.
	restored, err := tq.DeserializeFrom(&buf)
	if err != nil {
		t.Fatalf("DeserializeFrom: %v", err)
	}

	// Verify round-trip equality.
	if qv.Norm != restored.Norm {
		t.Errorf("norm mismatch: got %v, want %v", restored.Norm, qv.Norm)
	}
	if len(qv.Indices) != len(restored.Indices) {
		t.Fatalf("indices length mismatch: got %d, want %d", len(restored.Indices), len(qv.Indices))
	}
	for i := range qv.Indices {
		if qv.Indices[i] != restored.Indices[i] {
			t.Errorf("index[%d] mismatch: got %d, want %d", i, restored.Indices[i], qv.Indices[i])
		}
	}
}

// TestSerializeTo_DeserializeFrom_AllBitWidths tests streaming round-trip for all bit widths.
func TestSerializeTo_DeserializeFrom_AllBitWidths(t *testing.T) {
	bitWidths := []int{Bit2, Bit3, Bit4}
	for _, bw := range bitWidths {
		t.Run(bitWidthName(bw), func(t *testing.T) {
			tq, err := NewTurboQuant(128, bw, 99)
			if err != nil {
				t.Fatalf("NewTurboQuant: %v", err)
			}

			rng := rand.New(rand.NewSource(456))
			vec := make([]float32, 128)
			for i := range vec {
				vec[i] = rng.Float32()*10 - 5
			}

			qv, err := tq.Quantize(vec)
			if err != nil {
				t.Fatalf("Quantize: %v", err)
			}

			var buf bytes.Buffer
			if err := tq.SerializeTo(qv, &buf); err != nil {
				t.Fatalf("SerializeTo: %v", err)
			}

			restored, err := tq.DeserializeFrom(&buf)
			if err != nil {
				t.Fatalf("DeserializeFrom: %v", err)
			}

			if qv.Norm != restored.Norm {
				t.Errorf("norm mismatch: got %v, want %v", restored.Norm, qv.Norm)
			}
			if len(qv.Indices) != len(restored.Indices) {
				t.Fatalf("indices length mismatch: got %d, want %d", len(restored.Indices), len(qv.Indices))
			}
			for i := range qv.Indices {
				if qv.Indices[i] != restored.Indices[i] {
					t.Errorf("index[%d] mismatch: got %d, want %d", i, restored.Indices[i], qv.Indices[i])
				}
			}
		})
	}
}

// TestSerializeBatchTo_DeserializeBatchFrom_RoundTrip verifies batch round-trip
// via SerializeBatchTo/DeserializeBatchFrom using a bytes.Buffer.
func TestSerializeBatchTo_DeserializeBatchFrom_RoundTrip(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit3, 77)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(789))
	const batchSize = 10
	vecs := make([][]float32, batchSize)
	for i := range vecs {
		vecs[i] = make([]float32, 64)
		for j := range vecs[i] {
			vecs[i][j] = rng.Float32()*4 - 2
		}
	}

	qvs, err := tq.QuantizeBatch(vecs)
	if err != nil {
		t.Fatalf("QuantizeBatch: %v", err)
	}

	// Serialize batch to buffer.
	var buf bytes.Buffer
	if err := tq.SerializeBatchTo(qvs, &buf); err != nil {
		t.Fatalf("SerializeBatchTo: %v", err)
	}

	// Deserialize batch from buffer.
	restored, err := tq.DeserializeBatchFrom(&buf)
	if err != nil {
		t.Fatalf("DeserializeBatchFrom: %v", err)
	}

	if len(restored) != batchSize {
		t.Fatalf("batch size mismatch: got %d, want %d", len(restored), batchSize)
	}

	for i := range qvs {
		if qvs[i].Norm != restored[i].Norm {
			t.Errorf("vector[%d] norm mismatch: got %v, want %v", i, restored[i].Norm, qvs[i].Norm)
		}
		if len(qvs[i].Indices) != len(restored[i].Indices) {
			t.Fatalf("vector[%d] indices length mismatch: got %d, want %d", i, len(restored[i].Indices), len(qvs[i].Indices))
		}
		for j := range qvs[i].Indices {
			if qvs[i].Indices[j] != restored[i].Indices[j] {
				t.Errorf("vector[%d] index[%d] mismatch: got %d, want %d", i, j, restored[i].Indices[j], qvs[i].Indices[j])
			}
		}
	}
}

// TestSerializeBatchTo_EmptyBatch verifies that an empty batch round-trips correctly.
func TestSerializeBatchTo_EmptyBatch(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	var buf bytes.Buffer
	if err := tq.SerializeBatchTo(nil, &buf); err != nil {
		t.Fatalf("SerializeBatchTo: %v", err)
	}

	restored, err := tq.DeserializeBatchFrom(&buf)
	if err != nil {
		t.Fatalf("DeserializeBatchFrom: %v", err)
	}

	if len(restored) != 0 {
		t.Errorf("expected empty batch, got %d vectors", len(restored))
	}
}

// TestDeserializeFrom_TruncatedStream verifies error on truncated input.
func TestDeserializeFrom_TruncatedStream(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	// Only 2 bytes — not enough for even the norm header.
	buf := bytes.NewReader([]byte{0x00, 0x01})
	_, err = tq.DeserializeFrom(buf)
	if err == nil {
		t.Error("expected error for truncated stream, got nil")
	}
}

// TestDeserializeBatchFrom_TruncatedCount verifies error when count header is truncated.
func TestDeserializeBatchFrom_TruncatedCount(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	// Only 2 bytes — not enough for the 4-byte count header.
	buf := bytes.NewReader([]byte{0x01, 0x00})
	_, err = tq.DeserializeBatchFrom(buf)
	if err == nil {
		t.Error("expected error for truncated count header, got nil")
	}
}

// TestDeserializeBatchFrom_TruncatedVector verifies error when stream ends mid-vector.
func TestDeserializeBatchFrom_TruncatedVector(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	// Write count=1 but provide no vector data.
	var data bytes.Buffer
	countBuf := make([]byte, 4)
	countBuf[0] = 1 // count = 1, little-endian
	data.Write(countBuf)

	_, err = tq.DeserializeBatchFrom(&data)
	if err == nil {
		t.Error("expected error for truncated vector data, got nil")
	}
}

// TestStreamSerialize_FileIO verifies round-trip through actual file I/O.
func TestStreamSerialize_FileIO(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(321))
	vec := make([]float32, 64)
	for i := range vec {
		vec[i] = rng.Float32()*6 - 3
	}

	qv, err := tq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Write to temp file.
	tmpDir := t.TempDir()
	fpath := filepath.Join(tmpDir, "test_vector.bin")

	f, err := os.Create(fpath)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}
	if err := tq.SerializeTo(qv, f); err != nil {
		f.Close()
		t.Fatalf("SerializeTo file: %v", err)
	}
	f.Close()

	// Read from file.
	f, err = os.Open(fpath)
	if err != nil {
		t.Fatalf("os.Open: %v", err)
	}
	defer f.Close()

	restored, err := tq.DeserializeFrom(f)
	if err != nil {
		t.Fatalf("DeserializeFrom file: %v", err)
	}

	if qv.Norm != restored.Norm {
		t.Errorf("norm mismatch: got %v, want %v", restored.Norm, qv.Norm)
	}
	for i := range qv.Indices {
		if qv.Indices[i] != restored.Indices[i] {
			t.Errorf("index[%d] mismatch: got %d, want %d", i, restored.Indices[i], qv.Indices[i])
		}
	}
}

// TestStreamSerializeBatch_FileIO verifies batch round-trip through actual file I/O.
func TestStreamSerializeBatch_FileIO(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit3, 55)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(654))
	const batchSize = 5
	vecs := make([][]float32, batchSize)
	for i := range vecs {
		vecs[i] = make([]float32, 64)
		for j := range vecs[i] {
			vecs[i][j] = rng.Float32()*8 - 4
		}
	}

	qvs, err := tq.QuantizeBatch(vecs)
	if err != nil {
		t.Fatalf("QuantizeBatch: %v", err)
	}

	tmpDir := t.TempDir()
	fpath := filepath.Join(tmpDir, "test_batch.bin")

	f, err := os.Create(fpath)
	if err != nil {
		t.Fatalf("os.Create: %v", err)
	}
	if err := tq.SerializeBatchTo(qvs, f); err != nil {
		f.Close()
		t.Fatalf("SerializeBatchTo file: %v", err)
	}
	f.Close()

	f, err = os.Open(fpath)
	if err != nil {
		t.Fatalf("os.Open: %v", err)
	}
	defer f.Close()

	restored, err := tq.DeserializeBatchFrom(f)
	if err != nil {
		t.Fatalf("DeserializeBatchFrom file: %v", err)
	}

	if len(restored) != batchSize {
		t.Fatalf("batch size mismatch: got %d, want %d", len(restored), batchSize)
	}

	for i := range qvs {
		if qvs[i].Norm != restored[i].Norm {
			t.Errorf("vector[%d] norm mismatch", i)
		}
		for j := range qvs[i].Indices {
			if qvs[i].Indices[j] != restored[i].Indices[j] {
				t.Errorf("vector[%d] index[%d] mismatch", i, j)
			}
		}
	}
}

// TestSerializeTo_ConsistentWithSerialize verifies that SerializeTo produces
// the same bytes as the existing Serialize method.
func TestSerializeTo_ConsistentWithSerialize(t *testing.T) {
	tq, err := NewTurboQuant(64, Bit4, 42)
	if err != nil {
		t.Fatalf("NewTurboQuant: %v", err)
	}

	rng := rand.New(rand.NewSource(111))
	vec := make([]float32, 64)
	for i := range vec {
		vec[i] = rng.Float32()*2 - 1
	}

	qv, err := tq.Quantize(vec)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	// Serialize via existing method.
	expected, err := tq.Serialize(qv)
	if err != nil {
		t.Fatalf("Serialize: %v", err)
	}

	// Serialize via streaming method.
	var buf bytes.Buffer
	if err := tq.SerializeTo(qv, &buf); err != nil {
		t.Fatalf("SerializeTo: %v", err)
	}

	if !bytes.Equal(expected, buf.Bytes()) {
		t.Errorf("SerializeTo output differs from Serialize: got %d bytes, want %d bytes", buf.Len(), len(expected))
	}
}

func bitWidthName(bw int) string {
	switch bw {
	case Bit2:
		return "2bit"
	case Bit3:
		return "3bit"
	case Bit4:
		return "4bit"
	default:
		return "unknown"
	}
}
