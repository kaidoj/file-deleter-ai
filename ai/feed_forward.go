package ai

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// FeedForward calculates prediction
func FeedForward(m *Model, ctx *Context) *Context {
	wlen := len(m.weights) - 1 // skip output layer
	in := ctx.inputs.T()
	ctx.predictions = nil
	ctx.errors = nil

	// hidden layers
	for i := 0; i < wlen; i++ {
		in = activation(m, i, in)
		ctx.predictions = append(ctx.predictions, in)
	}

	// output layer
	for i := wlen; i < wlen+1; i++ {
		in = activation(m, i, in)
		ctx.predictions = append(ctx.predictions, in)
		ctx.prediction = in
	}

	// calc error
	s := new(mat.Dense)
	s.Sub(ctx.target, in)

	MatPrint(s)

	ctx.errors = s

	return ctx
}

func activation(m *Model, i int, in mat.Matrix) mat.Matrix {
	w := m.weights[i]
	b := m.biases[i].T()

	fmt.Println("weights")
	MatPrint(w)
	fmt.Println("inputs")
	MatPrint(in)

	dot := new(mat.Dense)
	dot.Mul(w, in)

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
