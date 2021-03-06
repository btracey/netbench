// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

import "errors"

func BatchPredict(batch BatchPredictor, inputs RowMatrix, outputs MutableRowMatrix,
	inputDim, outputDim int, grainSize int) (MutableRowMatrix, error) {

	// TODO: Add in something about error

	// Check that the inputs and outputs are the right sizes
	nSamples, dimInputs := inputs.Dims()
	if inputDim != dimInputs {
		return outputs, errors.New("predict batch: input dimension mismatch")
	}

	if outputs == nil {
		// In the real code this would be cause the creation of a mat64.Dense, but
		// for this benchmark suite we wish to avoid the mat64 dependency
		s := make(SosMatrix, nSamples)
		for i := range s {
			s[i] = make([]float64, outputDim)
		}
		outputs = s
	} else {
		nOutputSamples, dimOutputs := outputs.Dims()
		if dimOutputs != outputDim {
			return outputs, errors.New("predict batch: output dimension mismatch")
		}
		if nSamples != nOutputSamples {
			return outputs, errors.New("predict batch: rows mismatch")
		}
	}

	// Perform predictions in parallel. For each parallel call, form a new predictor so that
	// memory allocations are saved and no race condition happens.

	// If the input and/or output is a RowViewer, save time by avoiding a copy
	inputRVer, inputIsRowViewer := inputs.(RowViewer)
	outputRVer, outputIsRowViewer := outputs.(RowViewer)

	var f func(start, end int)

	// wrapper function to allow parallel prediction. Uses RowView if the type has it
	switch {
	default:
		panic("Shouldn't be here")
	case inputIsRowViewer, outputIsRowViewer:
		f = func(start, end int) {
			p := batch.NewPredictor()
			for i := start; i < end; i++ {
				p.Predict(inputRVer.RowView(i), outputRVer.RowView(i))
			}
		}

	case inputIsRowViewer && !outputIsRowViewer:
		f = func(start, end int) {
			p := batch.NewPredictor()
			output := make([]float64, outputDim)
			for i := start; i < end; i++ {
				outputs.Row(output, i)
				p.Predict(inputRVer.RowView(i), output)
				outputs.SetRow(i, output)
			}
		}
	case !inputIsRowViewer && outputIsRowViewer:
		f = func(start, end int) {
			p := batch.NewPredictor()
			input := make([]float64, inputDim)
			for i := start; i < end; i++ {
				inputs.Row(input, i)
				p.Predict(input, outputRVer.RowView(i))
			}
		}
	case !inputIsRowViewer && !outputIsRowViewer:
		f = func(start, end int) {
			p := batch.NewPredictor()
			input := make([]float64, inputDim)
			output := make([]float64, outputDim)
			for i := start; i < end; i++ {
				inputs.Row(input, i)
				outputs.Row(output, i)
				p.Predict(input, output)
				outputs.SetRow(i, output)
			}
		}
	}

	ParallelFor(nSamples, grainSize, f)
	return outputs, nil
}
