package turboquant_test

import (
	"fmt"
	"log"

	"github.com/mredencom/turboquant"
)

// ExampleNewTurboQuant demonstrates creating a TurboQuant quantizer instance.
func ExampleNewTurboQuant() {
	tq, err := turboquant.NewTurboQuant(8, 4, 42) // dim=8, 4-bit, seed=42
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Dimension: %d\n", tq.Dimension())
	fmt.Printf("BitWidth: %d\n", tq.BitWidth())
	fmt.Printf("CompressionRatio: %.2f\n", tq.CompressionRatio())
	// Output:
	// Dimension: 8
	// BitWidth: 4
	// CompressionRatio: 4.00
}

// ExampleTurboQuant_Quantize demonstrates quantizing a float32 vector.
func ExampleTurboQuant_Quantize() {
	tq, err := turboquant.NewTurboQuant(8, 4, 42)
	if err != nil {
		log.Fatal(err)
	}

	vec := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	qv, err := tq.Quantize(vec)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Norm: %.4f\n", qv.Norm)
	fmt.Printf("Indices length: %d\n", len(qv.Indices))
	// Output:
	// Norm: 14.2829
	// Indices length: 8
}

// ExampleTurboQuant_Serialize demonstrates the full serialize/deserialize round-trip.
func ExampleTurboQuant_Serialize() {
	tq, err := turboquant.NewTurboQuant(8, 4, 42)
	if err != nil {
		log.Fatal(err)
	}

	vec := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	qv, err := tq.Quantize(vec)
	if err != nil {
		log.Fatal(err)
	}

	// Serialize to compact binary
	data, err := tq.Serialize(qv)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Serialized bytes: %d\n", len(data))

	// Deserialize back
	qv2, err := tq.Deserialize(data)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Deserialized norm: %.4f\n", qv2.Norm)
	fmt.Printf("Indices match: %v\n", indicesEqual(qv.Indices, qv2.Indices))
	// Output:
	// Serialized bytes: 8
	// Deserialized norm: 14.2829
	// Indices match: true
}

// ExampleCosineSimilarity demonstrates computing cosine similarity
// between an original vector and its quantized-then-dequantized version.
func ExampleCosineSimilarity() {
	tq, err := turboquant.NewTurboQuant(8, 4, 42)
	if err != nil {
		log.Fatal(err)
	}

	vec := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	qv, err := tq.Quantize(vec)
	if err != nil {
		log.Fatal(err)
	}
	restored, err := tq.Dequantize(qv)
	if err != nil {
		log.Fatal(err)
	}

	sim, err := turboquant.CosineSimilarity(vec, restored)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Cosine similarity: %.4f\n", sim)
	// Output:
	// Cosine similarity: 0.9979
}

func indicesEqual(a, b []uint8) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
