// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

type Rower interface {
	Row([]float64, int) []float64
}

type Matrix interface {
	Dims() (r, c int)
	At(i, j int) float64
}

type Mutable interface {
	Matrix
	Set(i, j int, v float64)
}

type RowMatrix interface {
	Matrix
	Rower
}

type MutableRowMatrix interface {
	Rower
	Mutable
	SetRow(int, []float64) int
}

// A RowViewer can return a slice of float64 reflecting a row that is backed by the matrix
// data.
type RowViewer interface {
	RowView(r int) []float64
}

type BatchPredictor interface {
	NewPredictor() Predictor // Returns a predictor. This exists so that methods can create temporary data if necessary
}

// See package reggo for description. This is here to avoid circular imports
type Predictor interface {
	Predict(input, output []float64) ([]float64, error)
	PredictBatch(inputs RowMatrix, outputs MutableRowMatrix) (MutableRowMatrix, error)
	InputDim() int
	OutputDim() int
}

type Featurizer interface {
	// Featurize transforms the input into the elements of the feature matrix. Feature
	// will have length NumFeatures(). Should not modify input
	Featurize(input, feature []float64)
}

type LossDeriver interface {
	// Gets the current parameters
	//Parameters() []float64

	// Sets the current parameters
	//SetParameters([]float64)

	// Features is either the input or the output from Featurize
	// Deriv will be called after predict so memory may be cached
	Predict(parameters, featurizedInput, predOutput []float64)

	// Deriv computes the derivative of the loss with respect
	// to the weight given the predicted output and the derivative
	// of the loss function with respect to the prediction
	Deriv(parameters, featurizedInput, predOutput, dLossDPred, dLossDWeight []float64)
}
