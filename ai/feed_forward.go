package ai

import "math"

// FeedForward calculates prediction
func FeedForward(m *Model, ctx *Context) *Context {
	ctx.hiddenInputs = Dot(m.Inputs, m.weights.T())
	ctx.hiddenPredictions = Apply(calcSigmoid, ctx.hiddenInputs)
	ctx.outputs = Dot(ctx.hiddenPredictions, m.outputWeights.T())
	ctx.outputPredictions = Apply(calcSigmoid, ctx.outputs)
	ctx.OutputErrors = Substract(m.Outputs, ctx.outputPredictions)
	//ctx.outputErrors = Apply(calcAbs, ctx.OutputErrors)
	ctx.hiddenErrors = Dot(ctx.OutputErrors, m.outputWeights)
	return ctx
}

func calcSigmoid(_, _ int, v float64) float64 {
	return Sigmoid(v)
}

func calcSquare(_, _ int, v float64) float64 {
	return math.Abs(v) * math.Abs(v)
}

func calcAbs(_, _ int, v float64) float64 {
	return math.Abs(v)
}
