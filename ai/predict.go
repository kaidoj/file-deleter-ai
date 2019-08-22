package ai

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Predict(m *Model, ctx *Context) mat.Matrix {
	ctx.hiddenInputs = Dot(m.Inputs, m.weights.T())
	ctx.hiddenPredictions = Apply(calcSigmoid, ctx.hiddenInputs)
	ctx.outputs = Dot(ctx.hiddenPredictions, m.outputWeights.T())
	ctx.outputPredictions = Apply(calcSigmoid, ctx.outputs)
	MatPrint(ctx.outputPredictions)
	fmt.Println("predictions")
	errors := Substract(m.Outputs, ctx.outputPredictions)
	return errors
}
