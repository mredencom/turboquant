package turboquant

import (
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkQuantize(b *testing.B) {
	dimensions := []int{128, 256, 512, 1024}
	bitWidths := []int{2, 3, 4}

	for _, dim := range dimensions {
		for _, bw := range bitWidths {
			name := fmt.Sprintf("dim%d_bit%d", dim, bw)
			b.Run(name, func(b *testing.B) {
				tq, err := NewTurboQuant(dim, bw, 42)
				if err != nil {
					b.Fatalf("NewTurboQuant: %v", err)
				}

				rng := rand.New(rand.NewSource(99))
				vec := make([]float32, dim)
				for i := range vec {
					vec[i] = rng.Float32()*2 - 1
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, err := tq.Quantize(vec)
					if err != nil {
						b.Fatalf("Quantize: %v", err)
					}
				}
			})
		}
	}
}

func BenchmarkDequantize(b *testing.B) {
	dimensions := []int{128, 256, 512, 1024}
	bitWidths := []int{2, 3, 4}

	for _, dim := range dimensions {
		for _, bw := range bitWidths {
			name := fmt.Sprintf("dim%d_bit%d", dim, bw)
			b.Run(name, func(b *testing.B) {
				tq, err := NewTurboQuant(dim, bw, 42)
				if err != nil {
					b.Fatalf("NewTurboQuant: %v", err)
				}

				rng := rand.New(rand.NewSource(99))
				vec := make([]float32, dim)
				for i := range vec {
					vec[i] = rng.Float32()*2 - 1
				}

				qv, err := tq.Quantize(vec)
				if err != nil {
					b.Fatalf("Quantize: %v", err)
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, err := tq.Dequantize(qv)
					if err != nil {
						b.Fatalf("Dequantize: %v", err)
					}
				}
			})
		}
	}
}

func BenchmarkQuantizeBatch(b *testing.B) {
	batchSizes := []int{100, 1000, 10000}
	dim := 256
	bw := 4

	for _, bs := range batchSizes {
		name := fmt.Sprintf("batch%d_dim%d_bit%d", bs, dim, bw)
		b.Run(name, func(b *testing.B) {
			tq, err := NewTurboQuant(dim, bw, 42)
			if err != nil {
				b.Fatalf("NewTurboQuant: %v", err)
			}

			rng := rand.New(rand.NewSource(99))
			vecs := make([][]float32, bs)
			for i := range vecs {
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = rng.Float32()*2 - 1
				}
				vecs[i] = vec
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.QuantizeBatch(vecs)
				if err != nil {
					b.Fatalf("QuantizeBatch: %v", err)
				}
			}
		})
	}
}

func BenchmarkSerialize(b *testing.B) {
	bitWidths := []int{2, 3, 4}
	dim := 256

	for _, bw := range bitWidths {
		name := fmt.Sprintf("dim%d_bit%d", dim, bw)
		b.Run(name, func(b *testing.B) {
			tq, err := NewTurboQuant(dim, bw, 42)
			if err != nil {
				b.Fatalf("NewTurboQuant: %v", err)
			}

			rng := rand.New(rand.NewSource(99))
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = rng.Float32()*2 - 1
			}

			qv, err := tq.Quantize(vec)
			if err != nil {
				b.Fatalf("Quantize: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.Serialize(qv)
				if err != nil {
					b.Fatalf("Serialize: %v", err)
				}
			}
		})
	}
}

func BenchmarkDeserialize(b *testing.B) {
	bitWidths := []int{2, 3, 4}
	dim := 256

	for _, bw := range bitWidths {
		name := fmt.Sprintf("dim%d_bit%d", dim, bw)
		b.Run(name, func(b *testing.B) {
			tq, err := NewTurboQuant(dim, bw, 42)
			if err != nil {
				b.Fatalf("NewTurboQuant: %v", err)
			}

			rng := rand.New(rand.NewSource(99))
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = rng.Float32()*2 - 1
			}

			qv, err := tq.Quantize(vec)
			if err != nil {
				b.Fatalf("Quantize: %v", err)
			}

			data, err := tq.Serialize(qv)
			if err != nil {
				b.Fatalf("Serialize: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.Deserialize(data)
				if err != nil {
					b.Fatalf("Deserialize: %v", err)
				}
			}
		})
	}
}

func BenchmarkCodebookBuild(b *testing.B) {
	dimensions := []int{64, 128, 256}
	bitWidths := []int{2, 3, 4}

	for _, dim := range dimensions {
		for _, bw := range bitWidths {
			name := fmt.Sprintf("dim%d_bit%d", dim, bw)
			b.Run(name, func(b *testing.B) {
				builder := NewCodebookBuilder()

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, err := builder.Build(dim, bw)
					if err != nil {
						b.Fatalf("Build: %v", err)
					}
				}
			})
		}
	}
}

func BenchmarkNewRandomOrthogonalMatrix(b *testing.B) {
	dimensions := []int{64, 128, 256, 512}

	for _, dim := range dimensions {
		name := fmt.Sprintf("dim%d", dim)
		b.Run(name, func(b *testing.B) {
			var seed int64 = 42

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := NewRandomOrthogonalMatrix(dim, seed)
				if err != nil {
					b.Fatalf("NewRandomOrthogonalMatrix: %v", err)
				}
			}
		})
	}
}

// ---- Pool GC pressure benchmarks ----

func BenchmarkPoolGC_Quantize(b *testing.B) {
	dimensions := []int{128, 256, 512}
	bw := 4

	for _, dim := range dimensions {
		name := fmt.Sprintf("dim%d_bit%d", dim, bw)
		b.Run(name, func(b *testing.B) {
			tq, err := NewTurboQuant(dim, bw, 42)
			if err != nil {
				b.Fatalf("NewTurboQuant: %v", err)
			}

			rng := rand.New(rand.NewSource(99))
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = rng.Float32()*2 - 1
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.Quantize(vec)
				if err != nil {
					b.Fatalf("Quantize: %v", err)
				}
			}
		})
	}
}

func BenchmarkPoolGC_Dequantize(b *testing.B) {
	dimensions := []int{128, 256, 512}
	bw := 4

	for _, dim := range dimensions {
		name := fmt.Sprintf("dim%d_bit%d", dim, bw)
		b.Run(name, func(b *testing.B) {
			tq, err := NewTurboQuant(dim, bw, 42)
			if err != nil {
				b.Fatalf("NewTurboQuant: %v", err)
			}

			rng := rand.New(rand.NewSource(99))
			vec := make([]float32, dim)
			for i := range vec {
				vec[i] = rng.Float32()*2 - 1
			}

			qv, err := tq.Quantize(vec)
			if err != nil {
				b.Fatalf("Quantize: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.Dequantize(qv)
				if err != nil {
					b.Fatalf("Dequantize: %v", err)
				}
			}
		})
	}
}

func BenchmarkPoolGC_BatchQuantize(b *testing.B) {
	dim := 256
	bw := 4
	batchSize := 1000

	b.Run(fmt.Sprintf("batch%d_dim%d_bit%d", batchSize, dim, bw), func(b *testing.B) {
		tq, err := NewTurboQuant(dim, bw, 42)
		if err != nil {
			b.Fatalf("NewTurboQuant: %v", err)
		}

		rng := rand.New(rand.NewSource(99))
		vecs := make([][]float32, batchSize)
		for i := range vecs {
			vec := make([]float32, dim)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			vecs[i] = vec
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := tq.QuantizeBatch(vecs)
			if err != nil {
				b.Fatalf("QuantizeBatch: %v", err)
			}
		}
	})
}

// ---- Batch concurrency benchmarks ----

func BenchmarkBatchConcurrency(b *testing.B) {
	dim := 256
	bw := 4
	batchSize := 1000

	rng := rand.New(rand.NewSource(99))
	vecs := make([][]float32, batchSize)
	for i := range vecs {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()*2 - 1
		}
		vecs[i] = vec
	}

	concurrencyLevels := []struct {
		name string
		n    int
	}{
		{"serial_1", 1},
		{"workers_2", 2},
		{"workers_4", 4},
		{"workers_NumCPU", 0}, // 0 = default = runtime.NumCPU()
		{"unlimited_batch", batchSize},
	}

	for _, cl := range concurrencyLevels {
		b.Run(fmt.Sprintf("quantize/%s", cl.name), func(b *testing.B) {
			var opts []Option
			if cl.n > 0 {
				opts = append(opts, WithConcurrency(cl.n))
			}
			tq, err := NewTurboQuant(dim, bw, 42, opts...)
			if err != nil {
				b.Fatalf("NewTurboQuant: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.QuantizeBatch(vecs)
				if err != nil {
					b.Fatalf("QuantizeBatch: %v", err)
				}
			}
		})

		b.Run(fmt.Sprintf("dequantize/%s", cl.name), func(b *testing.B) {
			var opts []Option
			if cl.n > 0 {
				opts = append(opts, WithConcurrency(cl.n))
			}
			tq, err := NewTurboQuant(dim, bw, 42, opts...)
			if err != nil {
				b.Fatalf("NewTurboQuant: %v", err)
			}

			// Pre-quantize vectors for dequantize benchmark.
			qvs, err := tq.QuantizeBatch(vecs)
			if err != nil {
				b.Fatalf("QuantizeBatch: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := tq.DequantizeBatch(qvs)
				if err != nil {
					b.Fatalf("DequantizeBatch: %v", err)
				}
			}
		})
	}
}
