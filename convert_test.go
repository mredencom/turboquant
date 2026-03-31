package turboquant

import (
	"math"
	"testing"
)

func TestFloat64sToFloat32s(t *testing.T) {
	src := []float64{1.5, 2.5, 3.5}
	got := Float64sToFloat32s(src)
	if len(got) != 3 {
		t.Fatalf("expected len 3, got %d", len(got))
	}
	for i, v := range got {
		if math.Abs(float64(v)-src[i]) > 1e-6 {
			t.Errorf("index %d: got %f, want %f", i, v, src[i])
		}
	}
}

func TestFloat32sToFloat64s(t *testing.T) {
	src := []float32{1.5, 2.5, 3.5}
	got := Float32sToFloat64s(src)
	for i, v := range got {
		if v != float64(src[i]) {
			t.Errorf("index %d: got %f, want %f", i, v, float64(src[i]))
		}
	}
}

func TestIntsToFloat32s(t *testing.T) {
	src := []int{10, 20, -5}
	got := IntsToFloat32s(src)
	for i, v := range got {
		if int(v) != src[i] {
			t.Errorf("index %d: got %f, want %d", i, v, src[i])
		}
	}
}

func TestFloat32sToInts(t *testing.T) {
	src := []float32{10.4, 20.6, -5.5}
	got := Float32sToInts(src)
	want := []int{10, 21, -6}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("index %d: got %d, want %d", i, v, want[i])
		}
	}
}

func TestBytesToFloat32s(t *testing.T) {
	src := []byte{0, 128, 255}
	got := BytesToFloat32s(src)
	if got[0] != 0 || got[1] != 128 || got[2] != 255 {
		t.Errorf("got %v, want [0 128 255]", got)
	}
}

func TestFloat32sToBytes_Clamping(t *testing.T) {
	src := []float32{-10, 0, 128.4, 255.9, 300}
	got := Float32sToBytes(src)
	want := []byte{0, 0, 128, 255, 255}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("index %d: got %d, want %d", i, v, want[i])
		}
	}
}

func TestStringRoundTrip(t *testing.T) {
	s := "ABC"
	vec := StringToFloat32s(s)
	if len(vec) != 3 {
		t.Fatalf("expected len 3, got %d", len(vec))
	}
	// Direct round-trip without quantization should be exact
	back := Float32sToString(vec)
	if back != s {
		t.Errorf("got %q, want %q", back, s)
	}
}

func TestEmptySliceConversions(t *testing.T) {
	if len(Float64sToFloat32s(nil)) != 0 {
		t.Error("Float64sToFloat32s(nil) should return empty")
	}
	if len(Float32sToFloat64s(nil)) != 0 {
		t.Error("Float32sToFloat64s(nil) should return empty")
	}
	if len(IntsToFloat32s(nil)) != 0 {
		t.Error("IntsToFloat32s(nil) should return empty")
	}
	if len(Float32sToInts(nil)) != 0 {
		t.Error("Float32sToInts(nil) should return empty")
	}
	if len(BytesToFloat32s(nil)) != 0 {
		t.Error("BytesToFloat32s(nil) should return empty")
	}
	if len(Float32sToBytes(nil)) != 0 {
		t.Error("Float32sToBytes(nil) should return empty")
	}
	if StringToFloat32s("") != nil && len(StringToFloat32s("")) != 0 {
		t.Error("StringToFloat32s empty should return empty")
	}
	if Float32sToString(nil) != "" {
		t.Error("Float32sToString(nil) should return empty string")
	}
}
