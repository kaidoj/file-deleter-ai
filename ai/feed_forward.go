package ai

import (
	"gonum.org/v1/gonum/mat"
)

func FeedForward(m *Model, inp, targets mat.Matrix) mat.Matrix {

	wlen := len(m.weights) - 1 // skip output layer
	in := inp

	// hidden layers
	for i := 0; i < wlen; i++ {
		in = activition(m, i, in)
	}

	// output layer
	for i := wlen; i < wlen+1; i++ {
		in = activition(m, i, in)
	}

	// calc error
	s := new(mat.Dense)
	s.Sub(targets, in)

	return s
}

func activition(m *Model, i int, in mat.Matrix) mat.Matrix {
	w := m.weights[i]
	b := m.biases[i]

	dot := new(mat.Dense)
	dot.Mul(in, w)

	bz := new(mat.Dense)
	addBias := func(_, col int, v float64) float64 { return v + b.At(col, 0) }
	bz.Apply(addBias, dot)

	a := new(mat.Dense)
	a.Apply(calcSigmoid, bz)

	return a
}

func calcSigmoid(_, _ int, v float64) float64 {
	return Sigmoid(v)
}
