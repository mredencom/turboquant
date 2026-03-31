package main

import (
	"fmt"
	"strings"

	"github.com/mredencom/turboquant"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║         TurboQuant — Type Conversion Demo                   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("── 1. float64 vector ──")
	f64 := []float64{3.14159, 2.71828, 1.41421, 1.73205, 0.57721, 2.30258, 1.61803, 0.69314}
	demoQuantize(turboquant.Float64sToFloat32s(f64), 4, func(r []float32) {
		fmt.Printf("  Restored: %v\n  Original: %v\n", fmtF64(turboquant.Float32sToFloat64s(r)), fmtF64(f64))
	})

	fmt.Println("── 2. int slice ──")
	ints := []int{100, 200, 50, 75, 128, 255, 33, 180}
	demoQuantize(turboquant.IntsToFloat32s(ints), 4, func(r []float32) {
		fmt.Printf("  Restored: %v\n  Original: %v\n", turboquant.Float32sToInts(r), ints)
	})

	fmt.Println("── 3. byte slice ──")
	raw := []byte{0xFF, 0x00, 0xAB, 0xCD, 0x12, 0x34, 0x56, 0x78}
	demoQuantize(turboquant.BytesToFloat32s(raw), 4, func(r []float32) {
		fmt.Printf("  Restored: %X\n  Original: %X\n", turboquant.Float32sToBytes(r), raw)
	})

	fmt.Println("── 4. ASCII string ──")
	str := "TurboQt!"
	demoQuantize(turboquant.StringToFloat32s(str), 4, func(r []float32) {
		fmt.Printf("  Restored: %q\n  Original: %q\n", turboquant.Float32sToString(r), str)
	})

	fmt.Println("── 5. UTF-8 string (multi-byte) ──")
	utf8Str := "Hello, World! TurboQuant"
	utf8Bytes := []byte(utf8Str)
	fmt.Printf("  Original: %q (%d bytes)\n", utf8Str, len(utf8Bytes))
	tq, err := turboquant.NewTurboQuant(len(utf8Bytes), 4, 42)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}
	v := turboquant.BytesToFloat32s(utf8Bytes)
	qv, _ := tq.Quantize(v)
	restored, _ := tq.Dequantize(qv)
	rb := turboquant.Float32sToBytes(restored)
	fmt.Printf("  Restored: %q\n", string(rb))
	sim, _ := turboquant.CosineSimilarity(v, restored)
	fmt.Printf("  Cosine similarity: %.4f\n", sim)
	match := 0
	for i := range utf8Bytes {
		if utf8Bytes[i] == rb[i] {
			match++
		}
	}
	fmt.Printf("  Exact byte match: %d/%d (%.1f%%)\n\n", match, len(utf8Bytes), float64(match)/float64(len(utf8Bytes))*100)

	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("  Important Notes")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("  TurboQuant is lossy compression. Best suited for:")
	fmt.Println("  [Y] AI model embeddings — direction approximation, cosine sim > 0.98")
	fmt.Println("  [Y] LLM KV Cache — near-zero inference accuracy loss")
	fmt.Println("  [Y] Vector database search — approximate nearest neighbor")
	fmt.Println()
	fmt.Println("  Not suited for:")
	fmt.Println("  [N] Exact string/file recovery — lossy, no byte-level guarantee")
	fmt.Println("  [N] Encrypted data — quantization breaks data integrity")
	fmt.Println("  [N] Low-dimensional data (< 64 dim) — higher quantization error")
	fmt.Println()
}

func demoQuantize(vec []float32, bw int, show func([]float32)) {
	dim := len(vec)
	tq, err := turboquant.NewTurboQuant(dim, bw, 42)
	if err != nil {
		fmt.Printf("  ERROR: %v\n\n", err)
		return
	}
	qv, err := tq.Quantize(vec)
	if err != nil {
		fmt.Printf("  ERROR quantize: %v\n\n", err)
		return
	}
	data, _ := tq.Serialize(qv)
	restored, _ := tq.Dequantize(qv)
	sim, _ := turboquant.CosineSimilarity(vec, restored)
	fmt.Printf("  Original: %d bytes -> Serialized: %d bytes (%.1fx)\n", dim*4, len(data), float64(dim*4)/float64(len(data)))
	show(restored)
	fmt.Printf("  Cosine similarity: %.4f\n\n", sim)
}

func fmtF64(v []float64) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%.5f", f)
	}
	b.WriteByte(']')
	return b.String()
}
