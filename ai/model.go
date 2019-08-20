package ai

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	weights     []mat.Matrix
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
	model.biases = model.randomizeBias()

	return model
}

// Generate randomized values for matrix
func (m *Model) randomize() []mat.Matrix {
	var res []mat.Matrix
	lenL := len(m.layers) - 1
	for i := 0; i < lenL; i++ {
		l := m.layers[i]
		data := make([]float64, m.inputs*l)
		for j := range data {
			data[j] = rand.NormFloat64()
		}
		res = append(res, mat.NewDense(l, m.inputs, data))
	}

	//output layer
	data := make([]float64, m.layers[lenL-1])
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	res = append(res, mat.NewDense(1, m.layers[lenL-1], data))

	return res
}

// Generate randomized bias values for matrix
func (m *Model) randomizeBias() []*mat.Dense {
	var res []*mat.Dense
	for _, l := range m.layers {
		data := make([]float64, l)
		for i := range data {
			data[i] = rand.NormFloat64()
		}
		res = append(res, mat.NewDense(l, 1, data))
	}

	return res
}
