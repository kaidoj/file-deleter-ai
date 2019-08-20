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
	iRow, _ := model.Inputs.Dims()
	model.weights = model.randomize(model.HiddenNeurons, iRow)

	// generate random biases for hidden layers
	model.biases = model.randomize(1, model.HiddenNeurons)

	// generate random output weights
	oRow, _ := model.Outputs.Dims()
	model.outputWeights = model.randomize(oRow, model.HiddenNeurons)

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
