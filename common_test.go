// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

import (
	"math/rand"
	"testing"
)

const (
	throwPanic = true
	fdStep     = 1e-6
	fdTol      = 1e-6
)

func panics(f func()) (b bool) {
	defer func() {
		err := recover()
		if err != nil {
			b = true
		}
	}()
	f()
	return
}

func maybe(f func()) (b bool) {
	defer func() {
		err := recover()
		if err != nil {
			b = true
			if throwPanic {
				panic(err)
			}
		}
	}()
	f()
	return
}

type InputOutputer interface {
	InputDim() int
	OutputDim() int
}

func testInputOutputDim(t *testing.T, io InputOutputer, trueInputDim, trueOutputDim int, name string) {
	inputDim := io.InputDim()
	outputDim := io.OutputDim()
	if inputDim != trueInputDim {
		t.Errorf("%v: Mismatch in input dimension. expected %v, found %v", name, trueInputDim, inputDim)
	}
	if outputDim != trueOutputDim {
		t.Errorf("%v: Mismatch in input dimension. expected %v, found %v", name, trueOutputDim, inputDim)
	}
}

func RandomMat(r, c int, f func() float64) SosMatrix {
	s := make(SosMatrix, r)
	for i := range s {
		s[i] = make([]float64, c)
		for j := range s[i] {
			s[i][j] = f()
		}
	}
	return s
}

// TestPredict tests that predict returns the expected value, and that calling predict in parallel
// also works
func testPredictAndBatch(t *testing.T, p Predictor, inputs, trueOutputs RowMatrix, name string) {
	nSamples, inputDim := inputs.Dims()
	if inputDim != p.InputDim() {
		panic("input Dim doesn't match predictor input dim")
	}
	nOutSamples, outputDim := trueOutputs.Dims()
	if outputDim != p.OutputDim() {
		panic("outpuDim doesn't match predictor outputDim")
	}
	if nOutSamples != nSamples {
		panic("inputs and outputs have different number of rows")
	}

	// First, test sequentially
	for i := 0; i < nSamples; i++ {
		trueOut := make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			trueOut[j] = trueOutputs.At(i, j)
		}
		// Predict with nil
		input := make([]float64, inputDim)
		inputCpy := make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			input[j] = inputs.At(i, j)
			inputCpy[j] = inputs.At(i, j)
		}

		out1, err := p.Predict(input, nil)
		if err != nil {
			t.Errorf(name + ": Error predicting with nil output")
			return
		}
		if !Equal(input, inputCpy) {
			t.Errorf("%v: input changed with nil input for row %v", name, i)
			break
		}
		out2 := make([]float64, outputDim)
		for j := 0; j < outputDim; j++ {
			out2[j] = rand.NormFloat64()
		}

		_, err = p.Predict(input, out2)
		if err != nil {
			t.Errorf("%v: error predicting with non-nil input for row %v", name, i)
			break
		}
		if !Equal(input, inputCpy) {
			t.Errorf("%v: input changed with non-nil input for row %v", name, i)
			break
		}

		if !Equal(out1, out2) {
			t.Errorf(name + ": different answers with nil and non-nil predict ")
			break
		}
		if !EqualApprox(out1, trueOut, 1e-14) {
			t.Errorf("%v: predicted output doesn't match for row %v. Expected %v, found %v", name, i, trueOut, out1)
			break
		}
	}
}
