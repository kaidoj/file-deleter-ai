package ai

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type ModelConfig struct {
	LearingRate   float64
	Epochs        int
	HiddenNeurons int
	InputsCount   int
	OutputsCount  int
	Inputs        mat.Matrix
	Outputs       mat.Matrix
}

type Model struct {
	*ModelConfig
	weights       mat.Matrix
	outputWeights mat.Matrix
	biases        mat.Matrix
	outputBias    mat.Matrix
}

// NewModel starts new model with random weights
func NewModel(config *ModelConfig) *Model {
	model := &Model{
		config,
		nil,
		nil,
		nil,
		mat.NewDense(1, 1, []float64{1}),
	}

	// generate random weights for hidden layers
	model.weights = model.randomize(model.HiddenNeurons, model.InputsCount)

	// generate random biases for hidden layers
	model.biases = model.randomize(1, model.HiddenNeurons+1)

	// generate random output weights
	model.outputWeights = model.randomize(model.OutputsCount, model.HiddenNeurons)

	return model
}

// Generate randomized values for matrix
func (m *Model) randomize(rows, cols int) mat.Matrix {
	data := make([]float64, rows*cols)
	for j := range data {
		data[j] = rand.NormFloat64()
	}

	return mat.NewDense(rows, cols, data)
}
