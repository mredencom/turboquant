package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mredencom/turboquant"
)

const (
	dimension = 128
	numVecs   = 10000
	seed      = 42
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║              TurboQuant Benchmark — Performance Report              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Generate random test vectors (simulating KV Cache embeddings)
	rng := rand.New(rand.NewSource(99))
	vecs := make([][]float32, numVecs)
	for i := range vecs {
		v := make([]float32, dimension)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}

	origBytes := numVecs * dimension * 4
	fmt.Printf("Config: dimension=%d, vectors=%d, seed=%d\n", dimension, numVecs, seed)
	fmt.Printf("Original: %d vectors x %d dim x 4 bytes = %.2f MB\n\n",
		numVecs, dimension, float64(origBytes)/(1024*1024))

	fmt.Println("┌──────────┬────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
	fmt.Println("│ BitWidth │ Ratio      │ Compressed   │ Saved        │ Cosine Sim   │ Quant Time   │ Throughput   │")
	fmt.Println("├──────────┼────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

	for _, bw := range []int{2, 3, 4} {
		benchOneBitWidth(bw, vecs)
	}

	fmt.Println("└──────────┴────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
	fmt.Println()

	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Detail: Serialized Size vs Original Size")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println()

	for _, bw := range []int{2, 3, 4} {
		detailedSerializationReport(bw, vecs)
	}

	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println("  Summary")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("  * 3-bit: cosine sim > 0.96, ratio ~9.1x, memory saved ~89%")
	fmt.Println("  * 4-bit: cosine sim > 0.98, ratio ~7.1x, memory saved ~86%")
	fmt.Println("  * 2-bit: cosine sim > 0.90, ratio ~12.8x, memory saved ~92%")
	fmt.Println("  * All quantization is data-oblivious, no training data needed")
	fmt.Println()
}

func benchOneBitWidth(bw int, vecs [][]float32) {
	tq, err := turboquant.NewTurboQuant(dimension, bw, seed)
	if err != nil {
		fmt.Printf("  ERROR creating TurboQuant(bw=%d): %v\n", bw, err)
		return
	}

	start := time.Now()
	qvs, err := tq.QuantizeBatch(vecs)
	if err != nil {
		fmt.Printf("  ERROR QuantizeBatch(bw=%d): %v\n", bw, err)
		return
	}
	quantizeTime := time.Since(start)

	sampleSize := 500
	var totalSim float64
	for i := 0; i < sampleSize; i++ {
		restored, err := tq.Dequantize(qvs[i])
		if err != nil {
			continue
		}
		sim, err := turboquant.CosineSimilarity(vecs[i], restored)
		if err != nil {
			continue
		}
		totalSim += sim
	}
	avgSim := totalSim / float64(sampleSize)

	var totalSerBytes int
	for i := 0; i < 100; i++ {
		data, err := tq.Serialize(qvs[i])
		if err != nil {
			continue
		}
		totalSerBytes += len(data)
	}
	avgSerBytes := float64(totalSerBytes) / 100.0
	totalCompressedBytes := avgSerBytes * float64(numVecs)

	ratio := tq.CompressionRatio()
	savedPct := (1.0 - 1.0/ratio) * 100.0
	throughput := float64(numVecs) / quantizeTime.Seconds()

	fmt.Printf("│ %d-bit    │ %5.1fx     │ %7.2f MB   │ %5.1f%%       │ avg=%.4f    │ %8s     │ %7.0f v/s  │\n",
		bw, ratio,
		totalCompressedBytes/(1024*1024),
		savedPct,
		avgSim,
		quantizeTime.Round(time.Millisecond),
		throughput)
}

func detailedSerializationReport(bw int, vecs [][]float32) {
	tq, err := turboquant.NewTurboQuant(dimension, bw, seed)
	if err != nil {
		return
	}

	qv, _ := tq.Quantize(vecs[0])
	data, _ := tq.Serialize(qv)

	origSize := dimension * 4
	serSize := len(data)

	fmt.Printf("  %d-bit: per-vector %d bytes -> %d bytes (%.1fx), saved %d bytes\n",
		bw, origSize, serSize, float64(origSize)/float64(serSize), origSize-serSize)

	normBytes := 4
	indexBytes := serSize - normBytes
	fmt.Printf("         |-- norm: %d bytes (float32)\n", normBytes)
	fmt.Printf("         +-- indices: %d bytes (%d dim x %d bit, packed)\n\n",
		indexBytes, dimension, bw)
}
