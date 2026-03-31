package turboquant

import (
	"math"
	"math/rand"
	"sort"
	"testing"
	"testing/quick"
)

func TestFindNearestIndex(t *testing.T) {
	// Codebook with 4 centroids (2-bit) and 3 boundaries.
	// Intervals: (-inf, 0.25) -> 0, [0.25, 0.50) -> 1, [0.50, 0.75) -> 2, [0.75, +inf) -> 3
	cb := &Codebook{
		Centroids:  []float64{0.125, 0.375, 0.625, 0.875},
		Boundaries: []float64{0.25, 0.50, 0.75},
		BitWidth:   2,
	}

	tests := []struct {
		name  string
		value float64
		want  uint8
	}{
		{"below all boundaries", -1.0, 0},
		{"at first boundary", 0.25, 1},
		{"between first and second", 0.30, 1},
		{"at second boundary", 0.50, 2},
		{"between second and third", 0.60, 2},
		{"at third boundary", 0.75, 3},
		{"above all boundaries", 1.0, 3},
		{"exactly at centroid 0", 0.125, 0},
		{"exactly at centroid 3", 0.875, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cb.FindNearestIndex(tt.value)
			if got != tt.want {
				t.Errorf("FindNearestIndex(%v) = %d, want %d", tt.value, got, tt.want)
			}
		})
	}
}

func TestFindNearestIndex_SingleCentroid(t *testing.T) {
	// Edge case: codebook with a single centroid (no boundaries).
	cb := &Codebook{
		Centroids:  []float64{0.5},
		Boundaries: []float64{},
		BitWidth:   1,
	}

	if got := cb.FindNearestIndex(0.0); got != 0 {
		t.Errorf("FindNearestIndex(0.0) = %d, want 0", got)
	}
	if got := cb.FindNearestIndex(1.0); got != 0 {
		t.Errorf("FindNearestIndex(1.0) = %d, want 0", got)
	}
}

func TestFindNearestIndex_8Centroids(t *testing.T) {
	// 3-bit codebook: 8 centroids, 7 boundaries.
	cb := &Codebook{
		Centroids:  []float64{0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375},
		Boundaries: []float64{0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875},
		BitWidth:   3,
	}

	// Value in the last interval should return index 7.
	if got := cb.FindNearestIndex(0.9); got != 7 {
		t.Errorf("FindNearestIndex(0.9) = %d, want 7", got)
	}

	// Value in the first interval should return index 0.
	if got := cb.FindNearestIndex(0.1); got != 0 {
		t.Errorf("FindNearestIndex(0.1) = %d, want 0", got)
	}

	// Value at a middle boundary should return the next interval.
	if got := cb.FindNearestIndex(0.5); got != 4 {
		t.Errorf("FindNearestIndex(0.5) = %d, want 4", got)
	}
}

// --- CodebookBuilder Tests ---

func TestNewCodebookBuilder(t *testing.T) {
	builder := NewCodebookBuilder()
	if builder.gridPoints != 50000 {
		t.Errorf("expected gridPoints=50000, got %d", builder.gridPoints)
	}
	if builder.iterations != 300 {
		t.Errorf("expected iterations=300, got %d", builder.iterations)
	}
}

func TestBuild_InvalidBitWidth(t *testing.T) {
	builder := NewCodebookBuilder()
	for _, bw := range []int{0, 1, 5, 8} {
		_, err := builder.Build(64, bw)
		if err == nil {
			t.Errorf("Build(64, %d) expected error, got nil", bw)
		}
	}
}

func TestBuild_InvalidDimension(t *testing.T) {
	builder := NewCodebookBuilder()
	for _, dim := range []int{-1, 0, 1} {
		_, err := builder.Build(dim, 2)
		if err == nil {
			t.Errorf("Build(%d, 2) expected error, got nil", dim)
		}
	}
}

func TestBuild_CentroidCount(t *testing.T) {
	builder := NewCodebookBuilder()
	tests := []struct {
		dim, bw, wantCentroids int
	}{
		{64, 2, 4},
		{64, 3, 8},
		{64, 4, 16},
		{128, 2, 4},
	}
	for _, tt := range tests {
		cb, err := builder.Build(tt.dim, tt.bw)
		if err != nil {
			t.Fatalf("Build(%d, %d) unexpected error: %v", tt.dim, tt.bw, err)
		}
		if len(cb.Centroids) != tt.wantCentroids {
			t.Errorf("Build(%d, %d): got %d centroids, want %d", tt.dim, tt.bw, len(cb.Centroids), tt.wantCentroids)
		}
		if len(cb.Boundaries) != tt.wantCentroids-1 {
			t.Errorf("Build(%d, %d): got %d boundaries, want %d", tt.dim, tt.bw, len(cb.Boundaries), tt.wantCentroids-1)
		}
		if cb.BitWidth != tt.bw {
			t.Errorf("Build(%d, %d): BitWidth=%d, want %d", tt.dim, tt.bw, cb.BitWidth, tt.bw)
		}
	}
}

func TestBuild_CentroidsSorted(t *testing.T) {
	builder := NewCodebookBuilder()
	cb, err := builder.Build(64, 3)
	if err != nil {
		t.Fatalf("Build(64, 3) unexpected error: %v", err)
	}
	if !sort.Float64sAreSorted(cb.Centroids) {
		t.Error("centroids are not sorted ascending")
	}
}

func TestBuild_CentroidsInRange(t *testing.T) {
	builder := NewCodebookBuilder()
	cb, err := builder.Build(64, 4)
	if err != nil {
		t.Fatalf("Build(64, 4) unexpected error: %v", err)
	}
	for i, c := range cb.Centroids {
		if c < -1.0 || c > 1.0 {
			t.Errorf("centroid[%d]=%f is outside [-1, 1]", i, c)
		}
	}
}

func TestBuild_BoundariesBetweenCentroids(t *testing.T) {
	builder := NewCodebookBuilder()
	cb, err := builder.Build(64, 2)
	if err != nil {
		t.Fatalf("Build(64, 2) unexpected error: %v", err)
	}
	for i, b := range cb.Boundaries {
		if b <= cb.Centroids[i] || b >= cb.Centroids[i+1] {
			t.Errorf("boundary[%d]=%f not between centroid[%d]=%f and centroid[%d]=%f",
				i, b, i, cb.Centroids[i], i+1, cb.Centroids[i+1])
		}
	}
}

func TestBuild_SymmetricCentroids(t *testing.T) {
	// For symmetric Beta distribution, centroids should be approximately symmetric around 0.
	builder := NewCodebookBuilder()
	cb, err := builder.Build(64, 2)
	if err != nil {
		t.Fatalf("Build(64, 2) unexpected error: %v", err)
	}
	n := len(cb.Centroids)
	for i := 0; i < n/2; i++ {
		sum := cb.Centroids[i] + cb.Centroids[n-1-i]
		if math.Abs(sum) > 0.01 {
			t.Errorf("centroids not symmetric: c[%d]=%f + c[%d]=%f = %f",
				i, cb.Centroids[i], n-1-i, cb.Centroids[n-1-i], sum)
		}
	}
}

func TestBuild_SmallDimension(t *testing.T) {
	builder := NewCodebookBuilder()
	cb, err := builder.Build(2, 2)
	if err != nil {
		t.Fatalf("Build(2, 2) unexpected error: %v", err)
	}
	if len(cb.Centroids) != 4 {
		t.Errorf("expected 4 centroids, got %d", len(cb.Centroids))
	}
}

// --- Property-Based Tests ---

// TestProperty_CodebookCentroidCount verifies that for any valid (dimension, bitWidth),
// the built codebook has exactly 2^bitWidth centroids and 2^bitWidth - 1 boundaries.
//
// **Validates: Requirements 1.1, 1.6**
func TestProperty_CodebookCentroidCount(t *testing.T) {
	validBitWidths := []int{2, 3, 4}

	prop := func(seed uint32) bool {
		// Use seed to derive dimension and bitWidth deterministically.
		rng := rand.New(rand.NewSource(int64(seed)))

		// dimension in [2, 512]
		dimension := rng.Intn(511) + 2
		// bitWidth randomly picked from {2, 3, 4}
		bitWidth := validBitWidths[rng.Intn(len(validBitWidths))]

		// Use a lightweight builder with reduced grid/iterations for speed.
		builder := &CodebookBuilder{
			gridPoints: 5000,
			iterations: 30,
		}

		cb, err := builder.Build(dimension, bitWidth)
		if err != nil {
			t.Logf("Build(%d, %d) returned unexpected error: %v", dimension, bitWidth, err)
			return false
		}

		expectedCentroids := 1 << bitWidth
		expectedBoundaries := expectedCentroids - 1

		if len(cb.Centroids) != expectedCentroids {
			t.Logf("Build(%d, %d): got %d centroids, want %d", dimension, bitWidth, len(cb.Centroids), expectedCentroids)
			return false
		}
		if len(cb.Boundaries) != expectedBoundaries {
			t.Logf("Build(%d, %d): got %d boundaries, want %d", dimension, bitWidth, len(cb.Boundaries), expectedBoundaries)
			return false
		}
		if cb.BitWidth != bitWidth {
			t.Logf("Build(%d, %d): BitWidth=%d, want %d", dimension, bitWidth, cb.BitWidth, bitWidth)
			return false
		}

		return true
	}

	cfg := &quick.Config{MaxCount: 3}
	if err := quick.Check(prop, cfg); err != nil {
		t.Errorf("Property failed: %v", err)
	}
}

// --- Codebook Cache Tests ---

// TestGetOrBuildCodebook_SameParams verifies that calling GetOrBuildCodebook
// with the same (dimension, bitWidth) returns the exact same *Codebook pointer.
//
// Validates: Requirements 2.1
func TestGetOrBuildCodebook_SameParams(t *testing.T) {
	ResetCodebookCache()

	cb1, err := GetOrBuildCodebook(8, 2)
	if err != nil {
		t.Fatalf("first call: unexpected error: %v", err)
	}

	cb2, err := GetOrBuildCodebook(8, 2)
	if err != nil {
		t.Fatalf("second call: unexpected error: %v", err)
	}

	if cb1 != cb2 {
		t.Errorf("expected same pointer for same params, got different: %p vs %p", cb1, cb2)
	}
}

// TestGetOrBuildCodebook_DifferentParams verifies that calling GetOrBuildCodebook
// with different (dimension, bitWidth) returns different *Codebook instances.
//
// Validates: Requirements 2.2
func TestGetOrBuildCodebook_DifferentParams(t *testing.T) {
	ResetCodebookCache()

	cb1, err := GetOrBuildCodebook(8, 2)
	if err != nil {
		t.Fatalf("GetOrBuildCodebook(8, 2): unexpected error: %v", err)
	}

	cb2, err := GetOrBuildCodebook(8, 3)
	if err != nil {
		t.Fatalf("GetOrBuildCodebook(8, 3): unexpected error: %v", err)
	}

	cb3, err := GetOrBuildCodebook(16, 2)
	if err != nil {
		t.Fatalf("GetOrBuildCodebook(16, 2): unexpected error: %v", err)
	}

	if cb1 == cb2 {
		t.Error("expected different pointers for different bitWidth, got same")
	}
	if cb1 == cb3 {
		t.Error("expected different pointers for different dimension, got same")
	}
	if cb2 == cb3 {
		t.Error("expected different pointers for different (dimension, bitWidth), got same")
	}
}

// TestGetOrBuildCodebook_ConcurrentAccess verifies that concurrent calls to
// GetOrBuildCodebook with the same params all return the same pointer and
// do not panic.
//
// Validates: Requirements 2.3
func TestGetOrBuildCodebook_ConcurrentAccess(t *testing.T) {
	ResetCodebookCache()

	const goroutines = 16
	results := make(chan *Codebook, goroutines)
	errs := make(chan error, goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			cb, err := GetOrBuildCodebook(8, 2)
			errs <- err
			results <- cb
		}()
	}

	var first *Codebook
	for i := 0; i < goroutines; i++ {
		if err := <-errs; err != nil {
			t.Fatalf("goroutine %d: unexpected error: %v", i, err)
		}
		cb := <-results
		if first == nil {
			first = cb
		} else if cb != first {
			t.Errorf("goroutine %d: got different pointer %p, expected %p", i, cb, first)
		}
	}
}
