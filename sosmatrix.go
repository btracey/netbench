// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

// SOSMatrix is a Matrix wrapper around a slice of slice representation. It
// assumes that all of the "inner" slices have the same length
type SosMatrix [][]float64

func (s SosMatrix) Dims() (r, c int) {
	return len(s), len(s[0])
}

func (s SosMatrix) At(i, j int) float64 {
	return s[i][j]
}

func (s SosMatrix) Set(i, j int, v float64) {
	s[i][j] = v
}

func (s SosMatrix) RowView(r int) []float64 {
	return s[r]
}

func (s SosMatrix) Row(d []float64, i int) []float64 {
	if len(d) < len(s[0]) {
		d = make([]float64, len(s[0]))
	} else {
		d = d[:len(s[0])]
	}
	copy(d, s[i])
	return d
}

func (s SosMatrix) SetRow(i int, d []float64) int {
	return copy(s[i], d)
}
