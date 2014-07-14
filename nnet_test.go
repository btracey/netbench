// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

import (
	"math/rand"
	"strconv"
	"testing"
)

var testNets []*Trainer

var nSampleSlice []int = []int{1, 2, 3, 4, 5, 8, 16, 100, 102}

type netIniter struct {
	nHiddenLayers    int
	nNeuronsPerLayer int
	inputDim         int
	outputDim        int
	//nSamples int
	name                string
	finalLayerActivator Activator
}

var netIniters []*netIniter = []*netIniter{
	{
		nHiddenLayers:       0,
		nNeuronsPerLayer:    4,
		inputDim:            12,
		outputDim:           2,
		name:                "No hidden layers",
		finalLayerActivator: Linear{},
	},
	{
		nHiddenLayers:       2,
		nNeuronsPerLayer:    5,
		inputDim:            10,
		outputDim:           12,
		finalLayerActivator: Linear{},
	},
	{
		nHiddenLayers:       1,
		nNeuronsPerLayer:    5,
		inputDim:            10,
		outputDim:           8,
		finalLayerActivator: Linear{},
	},
	{
		nHiddenLayers:       3,
		nNeuronsPerLayer:    5,
		inputDim:            4,
		outputDim:           3,
		finalLayerActivator: Linear{},
	},
}

func init() {
	//runtime.GOMAXPROCS(runtime.NumCPU())
	for i, initer := range netIniters {
		n, err := NewSimpleTrainer(initer.inputDim, initer.outputDim, initer.nHiddenLayers, initer.nNeuronsPerLayer, initer.finalLayerActivator)
		if err != nil {
			panic(err)
		}
		if initer.name == "" {
			initer.name = strconv.Itoa(i)
		}
		n.RandomizeParameters()
		testNets = append(testNets, n)
	}
}

func TestInputOutputDim(t *testing.T) {
	for i, test := range netIniters {
		n := testNets[i]
		testInputOutputDim(t, n, test.inputDim, test.outputDim, test.name)
	}
}

const nRandSamp int = 1000

func TestPublicPredictAndBatch(t *testing.T) {
	for i, test := range netIniters {
		for _, nSamples := range nSampleSlice {
			n := testNets[i]
			inputs := RandomMat(nSamples, test.inputDim, rand.NormFloat64)
			trueOutputs := RandomMat(nSamples, test.outputDim, rand.NormFloat64)

			for j := 0; j < nSamples; j++ {
				tmp1, tmp2 := newPredictMemory(n.neurons)
				predict(inputs.RowView(j), n.neurons, n.parameters, tmp1, tmp2, trueOutputs.RowView(j))
				//predict(inputs.RowView(j), s.features, s.b, s.featureWeights, trueOutputs.RowView(j))
			}

			testPredictAndBatch(t, n, inputs, trueOutputs, test.name)
		}
	}
}
