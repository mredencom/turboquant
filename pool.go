package turboquant

import "sync"

// Slice pools for reducing GC pressure in hot quantization/dequantization paths.
// Pools are keyed by slice length (dimension) to avoid capacity mismatches.
//
// IMPORTANT: Only use these pools for temporary intermediate slices that are
// NOT returned to the caller. Slices like QuantizedVector.Indices or the
// final dequantized float32 result must NOT come from the pool.

var (
	float64SlicePool sync.Map // map[int]*sync.Pool — pools keyed by dimension
)

// getFloat64Pool returns the sync.Pool for []float64 slices of the given dimension.
func getFloat64Pool(dim int) *sync.Pool {
	if v, ok := float64SlicePool.Load(dim); ok {
		return v.(*sync.Pool)
	}
	pool := &sync.Pool{
		New: func() any {
			s := make([]float64, dim)
			return &s
		},
	}
	actual, _ := float64SlicePool.LoadOrStore(dim, pool)
	return actual.(*sync.Pool)
}

// getFloat64Slice retrieves a []float64 slice of the given dimension from the pool.
// The returned slice is zeroed before return.
func getFloat64Slice(dim int) []float64 {
	pool := getFloat64Pool(dim)
	sp := pool.Get().(*[]float64)
	s := *sp
	// Zero the slice to prevent stale data leaking between uses.
	clear(s)
	return s
}

// putFloat64Slice returns a []float64 slice to the pool.
// Only slices whose length matches the pool dimension should be returned.
func putFloat64Slice(s []float64) {
	if len(s) == 0 {
		return
	}
	pool := getFloat64Pool(len(s))
	pool.Put(&s)
}
