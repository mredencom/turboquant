package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mredencom/turboquant"
)

// Simulated LLM KV Cache parameters
const (
	numLayers = 32
	numHeads  = 8
	headDim   = 128
	seqLen    = 512
	bitWidth  = 4
	seed      = 42
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║       TurboQuant — LLM KV Cache Compression Demo           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	totalVectors := numLayers * numHeads * seqLen
	origBytes := totalVectors * headDim * 4 // float32 = 4 bytes

	fmt.Println("  Model Configuration")
	fmt.Printf("    Layers:          %d\n", numLayers)
	fmt.Printf("    Heads/layer:     %d\n", numHeads)
	fmt.Printf("    Head dimension:  %d\n", headDim)
	fmt.Printf("    Sequence length: %d\n", seqLen)
	fmt.Printf("    Total vectors:   %d (K) + %d (V) = %d\n", totalVectors, totalVectors, totalVectors*2)
	fmt.Printf("    Original memory: %.2f MB\n", float64(origBytes*2)/(1024*1024))
	fmt.Println()

	// Initialize quantizer
	fmt.Printf("  Initializing %d-bit quantizer (dim=%d) ...\n", bitWidth, headDim)
	initStart := time.Now()
	tq, err := turboquant.NewTurboQuant(headDim, bitWidth, seed)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}
	fmt.Printf("  Init time: %s\n\n", time.Since(initStart).Round(time.Millisecond))

	// Generate random KV cache vectors
	fmt.Println("  Generating random KV cache vectors ...")
	rng := rand.New(rand.NewSource(99))
	keys := generateVectors(rng, totalVectors, headDim)
	values := generateVectors(rng, totalVectors, headDim)
	fmt.Printf("  Generated %d key + %d value vectors\n\n", len(keys), len(values))

	// Quantize keys and values
	fmt.Println("  Quantizing ...")
	quantStart := time.Now()
	qKeys, err := tq.QuantizeBatch(keys)
	if err != nil {
		fmt.Printf("  ERROR quantizing keys: %v\n", err)
		return
	}
	qValues, err := tq.QuantizeBatch(values)
	if err != nil {
		fmt.Printf("  ERROR quantizing values: %v\n", err)
		return
	}
	quantTime := time.Since(quantStart)

	// Measure compressed size via serialization
	var compressedBytes int
	for _, qv := range qKeys {
		data, _ := tq.Serialize(qv)
		compressedBytes += len(data)
	}
	for _, qv := range qValues {
		data, _ := tq.Serialize(qv)
		compressedBytes += len(data)
	}

	ratio := tq.CompressionRatio()
	savedPct := (1.0 - 1.0/ratio) * 100.0
	throughput := float64(totalVectors*2) / quantTime.Seconds()

	// Verify quality: dequantize a sample and check cosine similarity
	sampleSize := 1000
	if sampleSize > totalVectors {
		sampleSize = totalVectors
	}
	var totalSim float64
	for i := 0; i < sampleSize; i++ {
		restored, err := tq.Dequantize(qKeys[i])
		if err != nil {
			continue
		}
		sim, err := turboquant.CosineSimilarity(keys[i], restored)
		if err != nil {
			continue
		}
		totalSim += sim
	}
	avgSim := totalSim / float64(sampleSize)

	// Print summary table
	fmt.Println()
	fmt.Println("┌──────────────────────────────┬──────────────────────────────┐")
	fmt.Println("│ Metric                       │ Value                        │")
	fmt.Println("├──────────────────────────────┼──────────────────────────────┤")
	fmt.Printf("│ Original memory              │ %8.2f MB                  │\n", float64(origBytes*2)/(1024*1024))
	fmt.Printf("│ Compressed memory            │ %8.2f MB                  │\n", float64(compressedBytes)/(1024*1024))
	fmt.Printf("│ Compression ratio            │ %8.1fx                    │\n", ratio)
	fmt.Printf("│ Memory saved                 │ %8.1f%%                    │\n", savedPct)
	fmt.Printf("│ Quantize time (all vectors)  │ %8s                     │\n", quantTime.Round(time.Millisecond))
	fmt.Printf("│ Throughput                   │ %8.0f vectors/s           │\n", throughput)
	fmt.Printf("│ Avg cosine similarity (keys) │ %8.6f                    │\n", avgSim)
	fmt.Printf("│ Bit width                    │ %8d-bit                  │\n", bitWidth)
	fmt.Println("└──────────────────────────────┴──────────────────────────────┘")
	fmt.Println()

	// Per-layer breakdown
	fmt.Println("  Per-Layer Memory Breakdown")
	vecsPerLayer := numHeads * seqLen
	layerOrigBytes := vecsPerLayer * headDim * 4 * 2 // K+V
	var layerCompBytes int
	for i := 0; i < vecsPerLayer; i++ {
		d1, _ := tq.Serialize(qKeys[i])
		d2, _ := tq.Serialize(qValues[i])
		layerCompBytes += len(d1) + len(d2)
	}
	fmt.Printf("    Layer 0: %.2f MB -> %.2f MB (%.1fx)\n",
		float64(layerOrigBytes)/(1024*1024),
		float64(layerCompBytes)/(1024*1024),
		float64(layerOrigBytes)/float64(layerCompBytes))
	fmt.Printf("    (all %d layers follow the same pattern)\n\n", numLayers)

	fmt.Println("  Conclusion")
	fmt.Printf("    With %d-bit TurboQuant, a %d-layer/%d-head KV cache\n", bitWidth, numLayers, numHeads)
	fmt.Printf("    at sequence length %d is compressed from %.1f MB to %.1f MB\n",
		seqLen, float64(origBytes*2)/(1024*1024), float64(compressedBytes)/(1024*1024))
	fmt.Printf("    with %.4f average cosine similarity — near-lossless for LLM inference.\n\n", avgSim)
}

func generateVectors(rng *rand.Rand, count, dim int) [][]float32 {
	vecs := make([][]float32, count)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}
	return vecs
}
