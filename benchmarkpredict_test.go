// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

import (
	"math/rand"
	"testing"
)

/*

About these tests:
These test the speed of execution of neural net predictions. The benchmark names
have 4 numbers. The numbers, in order, represent
	1) InputDim -- the dimension of the inputs
	2) NumLayers -- the number of hidden layers of the neural net
	3) NumNeuronsPerLayer -- how many neurons per hidden layer
	4) NumSamples -- how many different locations is the net predicting.
In all cases the number of outputs is 1

In a neural net, the "Neuron" is the basic computational unit. Each neuron makes
a combination of its inputs and then takes some activation function of its inputs
In the basic neural net, the combination is a weighted sum of the inputs and
the activation is a tanh function.

The number of inputs to layer n is the number of neurons in layer n - 1, where
layer 0 has NumInputs effective outputs. Thus, for a traditional net, the
computation time is O(nInputs * nNPL + (nL - 1)*nNPL^2 + nOutputs * nNPL)

This is a benchmark of numeric computation power, interface dereferencing, and
of the runtime to compute a pleasingly parallel loop.
*/

// This is a small net where there is relatively smaller amount of computation
// and a relatively larger amount of overhead
func BenchmarkPredictBatch_10_1_5_100000(t *testing.B) {
	benchmarkPredictBatch(t, 10, 1, 1, 5, 100000)
}

// This is a medium-sized net where there is more computatiot per sample
func BenchmarkPredictBatch_4_15_15_10000(t *testing.B) {
	benchmarkPredictBatch(t, 4, 1, 15, 15, 10000)
}

// This is a large net where there is a lot more computation per sample so the
// overhead should be relatively lower
func BenchmarkPredictBatch_100_10_50_1000(t *testing.B) {
	benchmarkPredictBatch(t, 100, 1, 10, 50, 1000)
}

// This net has a lot of small layers though a sizeable computation time overall
// The parallel overhead shouldn't be that great but the interface dereferencing
// might
func BenchmarkPredictBatch_10_100_2_10000(t *testing.B) {
	benchmarkPredictBatch(t, 10, 1, 100, 2, 10000)
}

// This has a lot of computation per neuron so should be more of a pure computation
// test
func BenchmarkPredictBatch_1000_1_10_10000(t *testing.B) {
	benchmarkPredictBatch(t, 1000, 1, 1, 10, 10000)
}

func benchmarkPredictBatch(b *testing.B, inputDim, outputDim, nLayers, nNeurons, nSamples int) {
	// Construct net
	trainer, err := NewSimpleTrainer(inputDim, outputDim, nLayers, nNeurons, Linear{})
	if err != nil {
		panic(err)
	}
	trainer.RandomizeParameters()
	//fmt.Println("Grain size is ", trainer.GrainSize(), "For input = ", inputDim, " nLayers = ", nLayers, " nNeuronsPerLayer = ", nNeurons)

	net := trainer.Predictor()

	input := RandomMat(nSamples, inputDim, rand.NormFloat64)

	output := RandomMat(nSamples, outputDim, rand.NormFloat64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		net.PredictBatch(input, output)
	}
}
