// Copyright 2013 The Gonum Authors. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

import "math"

func Equal(s, t []float64) bool {
	if len(s) != len(t) {
		return false
	}
	for i, v := range s {
		if v != t[i] {
			return false
		}
	}
	return true
}

// EqualsApprox returns true if the slices have equal lengths and
// all element pairs have an absolute tolerance less than tol or a
// relative tolerance less than tol.
func EqualApprox(s1, s2 []float64, tol float64) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, a := range s1 {
		if !EqualWithinAbsOrRel(a, s2[i], tol, tol) {
			return false
		}
	}
	return true
}

// EqualsFunc returns true if the slices have the same lengths
// and the function returns true for all element pairs.
func EqualFunc(s1, s2 []float64, f func(float64, float64) bool) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, val := range s1 {
		if !f(val, s2[i]) {
			return false
		}
	}
	return true

}

// ApproxEqualAbs returns true if a and b have an absolute
// difference of less than tol.
func EqualWithinAbs(a, b, tol float64) bool {
	return a == b || math.Abs(a-b) <= tol
}

const minNormalFloat64 = 2.2250738585072014e-308

// ApproxEqualRel returns true if the difference between a and b
// is not greater than tol times the greater value.
func EqualWithinRel(a, b, tol float64) bool {
	if a == b {
		return true
	}
	delta := math.Abs(a - b)
	if delta <= minNormalFloat64 {
		return delta <= tol*minNormalFloat64
	}
	// We depend on the division in this relationship to identify
	// infinities (we rely on the NaN to fail the test) otherwise
	// we compare Infs of the same sign and evaluate Infs as equal
	// independent of sign.
	return delta/math.Max(math.Abs(a), math.Abs(b)) <= tol
}

// EqualsWithinAbsOrRel returns true if a and b are equal to within
// the absolute tolerance.
func EqualWithinAbsOrRel(a, b, absTol, relTol float64) bool {
	if EqualWithinAbs(a, b, absTol) {
		return true
	}
	return EqualWithinRel(a, b, relTol)
}
