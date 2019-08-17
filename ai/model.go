package ai

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	weights     []*mat.Dense
	biases      []*mat.Dense
	layers      []int
	inputs      int
	outputs     int
	learingRate float64
	epochs      int
}

// NewModel creates new model
func NewModel(hl []int, inputs, outputs int, lr float64, e int) *Model {
	model := &Model{}
	model.layers = hl
	model.inputs = inputs
	model.outputs = outputs
	model.learingRate = lr
	model.epochs = e

	// generate random weights
	model.weights = model.randomize()

	// generate random biases
	model.biases = model.randomize()

	return model
}

// Generate randomized values for matrix
func (m *Model) randomize() []*mat.Dense {
	var res []*mat.Dense
	for _, l := range m.layers {
		data := make([]float64, m.inputs*l)
		for i := range data {
			data[i] = rand.NormFloat64()
		}
		res = append(res, mat.NewDense(m.inputs, l, data))
	}

	return res
}
