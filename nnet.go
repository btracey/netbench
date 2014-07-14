// Copyright 2013 Brendan Tracey. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file

package nnet

import (
	"errors"
	"math"
)

// Net is a simple feed-forward neural net
type Net struct {
	inputDim           int
	outputDim          int
	totalNumParameters int

	grainSize int

	neurons    [][]Neuron
	parameters [][][]float64
}

// InputDim returns the number of inputs expected by the net
func (n *Net) InputDim() int {
	return n.inputDim
}

// InputDim returns the number of outputs expected by Sink
func (n *Net) OutputDim() int {
	return n.outputDim
}

func (n *Net) Predict(input []float64, output []float64) ([]float64, error) {
	if len(input) != n.inputDim {
		return nil, errors.New("input dimension mismatch")
	}
	if output == nil {
		output = make([]float64, n.outputDim)
	} else {
		if len(output) != n.outputDim {
			return nil, errors.New("output dimension mismatch")
		}
	}
	prevOutput, tmpOutput := newPredictMemory(n.neurons)
	predict(input, n.neurons, n.parameters, prevOutput, tmpOutput, output)
	return output, nil
}

func (n *Net) PredictBatch(inputs RowMatrix, outputs MutableRowMatrix) (MutableRowMatrix, error) {
	batch := batchPredictor{
		neurons:    n.neurons,
		parameters: n.parameters,
		inputDim:   n.InputDim(),
		outputDim:  n.OutputDim(),
	}
	return BatchPredict(batch, inputs, outputs, n.inputDim, n.outputDim, n.grainSize)
}

func (n *Net) HackGrainSize(g int) {
	n.grainSize = g
}

func (n *Net) setGrainSize() {
	// The number of floating point operations is proportional to the number of
	// parameters, plus there is some overhead per neuron per function call and
	// some overhead per layer in function calls

	// Numbers determined unscientifically by using benchmark results. Definitely
	// dependent on many things, but these are probably good enough
	neuronOverhead := 70 // WAG, relative to one parameter
	layerOverhead := 200 // relative to one parameter

	var nNeurons int
	for _, layer := range n.neurons {
		nNeurons += len(layer)
	}

	nOps := n.totalNumParameters + nNeurons*neuronOverhead + layerOverhead*len(n.neurons)

	// We want each batch to take around 100µs
	// https://groups.google.com/forum/#!searchin/golang-nuts/Data$20parallelism$20with$20go$20routines/golang-nuts/-9LdBZoAIrk/2ayBvi0U0mQJ

	// Something like "nanoseconds per effective parameter"
	// Determined non-scientifically from benchmarks. This is definitely architecture
	// dependent, but maybe not relative to the overhead of the parallel loop
	c := 0.7

	grainSize := int(math.Ceil(100000 / (c * float64(nOps))))
	if grainSize < 1 {
		grainSize = 1 // This shouldn't happen, but maybe for a REALLY large net. Better safe than sorry
	}

	n.grainSize = grainSize
}

func (n *Net) GrainSize() int {
	return n.grainSize
}

// batchPredictor is a type which implements BatchPredictor so that
// the predictions can be computed in parallel
type batchPredictor struct {
	neurons    [][]Neuron
	parameters [][][]float64
	inputDim   int
	outputDim  int
}

// NewPredictor generates the necessary temporary memory and returns a struct to allow
// for concurrent prediction
func (b batchPredictor) NewPredictor() Predictor {
	prevOutput, tmpOutput := newPredictMemory(b.neurons)
	return predictor{
		neurons:       b.neurons,
		parameters:    b.parameters,
		tmpOutput:     tmpOutput,
		prevTmpOutput: prevOutput,
		inputDim:      b.inputDim,
		outputDim:     b.outputDim,
	}
}

// predictor is a struct that contains temporary memory to be reused during
// sucessive calls to predict
type predictor struct {
	neurons       [][]Neuron
	parameters    [][][]float64
	tmpOutput     []float64
	prevTmpOutput []float64

	inputDim  int
	outputDim int
}

func (p predictor) Predict(input, output []float64) ([]float64, error) {
	predict(input, p.neurons, p.parameters, p.prevTmpOutput, p.tmpOutput, output)
	return output, nil
}

// this is here because in the real code there are two different definitions
// of Predictor. One for internal use and one for external use.
func (p predictor) PredictBatch(RowMatrix, MutableRowMatrix) (MutableRowMatrix, error) {
	panic("can't be here")
}

func (p predictor) InputDim() int {
	return p.inputDim
}

func (p predictor) OutputDim() int {
	return p.outputDim
}

func newPredictMemory(neurons [][]Neuron) (prevOutput, output []float64) {
	// find the largest layer in terms of number of neurons
	max := len(neurons[0])
	for i := 1; i < len(neurons); i++ {
		l := len(neurons[i])
		if l > max {
			max = l
		}
	}
	return make([]float64, max), make([]float64, max)
}

// predict predicts the output from the net. prevOutput and output are
func predict(input []float64, neurons [][]Neuron, parameters [][][]float64, prevTmpOutput, tmpOutput []float64, output []float64) {
	nLayers := len(neurons)

	if nLayers == 1 {
		processLayer(input, neurons[0], parameters[0], output)
		return
	}

	// first layer uses the real input as the input
	tmpOutput = tmpOutput[:len(neurons[0])]
	processLayer(input, neurons[0], parameters[0], tmpOutput)

	// Middle layers use the previous output as input
	for i := 1; i < nLayers-1; i++ {
		// swap the pointers for temporary outputs, and make the new output the correct size
		prevTmpOutput, tmpOutput = tmpOutput, prevTmpOutput
		tmpOutput = tmpOutput[:len(neurons[i])]
		processLayer(prevTmpOutput, neurons[i], parameters[i], tmpOutput)
	}
	// The final layer is the actual output
	processLayer(tmpOutput, neurons[nLayers-1], parameters[nLayers-1], output)
}

func processLayer(input []float64, neurons []Neuron, parameters [][]float64, output []float64) {
	for i, neuron := range neurons {
		combination := neuron.Combine(parameters[i], input)
		output[i] = neuron.Activate(combination)
	}
}

// Trainer is a wrapper for the feed-forward net for training
type Trainer struct {
	*Net
}

// NewSimpleTrainer constructs a trainable feed-forward neural net with the specified sizes and
// tanh neuron activators in the hidden layer. Common choices for the final layer activator
// are activator.Linear for regression and activator.Tanh for classification.
// nLayers is the number of hidden layers. For now, must be at least one.
func NewSimpleTrainer(inputDim, outputDim, nHiddenLayers, nNeuronsPerLayer int, finalLayerActivator Activator) (*Trainer, error) {
	if inputDim <= 0 {
		return nil, errors.New("non-positive input dimension")
	}
	if outputDim <= 0 {
		return nil, errors.New("non-positive output dimension")
	}
	if inputDim <= 0 {
		return nil, errors.New("non-positive number of neurons per layer")
	}

	// Create the neurons
	// the hidden layers have the same number of neurons as hidden layers
	// final layer has a number of neurons equal to the number of outputs

	neurons := make([][]Neuron, nHiddenLayers+1)
	for i := 0; i < nHiddenLayers; i++ {
		neurons[i] = make([]Neuron, nNeuronsPerLayer)
		for j := 0; j < nNeuronsPerLayer; j++ {
			neurons[i][j] = TanhNeuron
		}
	}

	neurons[nHiddenLayers] = make([]Neuron, outputDim)
	for i := 0; i < outputDim; i++ {
		neurons[nHiddenLayers][i] = SumNeuron{Activator: finalLayerActivator}
	}
	return NewTrainer(inputDim, outputDim, neurons)
}

// NewTrainer creates a new feed-forward neural net with the given layers
func NewTrainer(inputDim, outputDim int, neurons [][]Neuron) (*Trainer, error) {
	if len(neurons) == 0 {
		return nil, errors.New("net: no neurons given")
	}
	for i := range neurons {
		if len(neurons[i]) == 0 {
			return nil, errors.New("net: layer with no neurons")
		}
	}

	// Create the parameters, the number of parameters, and the parameter index
	nLayers := len(neurons)
	parameters := make([][][]float64, nLayers)

	totalNumParameters := 0
	nLayerInputs := inputDim
	for i, layer := range neurons {
		parameters[i] = make([][]float64, len(layer))
		for j, neuron := range layer {
			nParameters := neuron.NumParameters(nLayerInputs)
			parameters[i][j] = make([]float64, nParameters)
			totalNumParameters += nParameters
		}
		nLayerInputs = len(layer)
	}
	net := &Net{
		inputDim:           inputDim,
		outputDim:          outputDim,
		totalNumParameters: totalNumParameters,
		neurons:            neurons,
		parameters:         parameters,
	}
	net.setGrainSize()
	return &Trainer{net}, nil
}

func newPerParameterMemory(params [][][]float64) [][][]float64 {
	n := make([][][]float64, len(params))
	for i, layer := range params {
		n[i] = make([][]float64, len(layer))
		for j, _ := range layer {
			n[i][j] = make([]float64, len(params[i][j]))
		}
	}
	return n
}

func newPerNeuronMemory(n [][]Neuron) [][]float64 {
	sos := make([][]float64, len(n))
	for i, layer := range n {
		sos[i] = make([]float64, len(layer))
	}
	return sos
}

// TODO: Replace this with a copy so can modify the trainer after releasing the
// predictor
func (s *Trainer) Predictor() Predictor {
	return s.Net
}

func (s *Trainer) RandomizeParameters() {
	for i, layer := range s.neurons {
		for j, neuron := range layer {
			neuron.Randomize(s.parameters[i][j])
		}
	}
}
