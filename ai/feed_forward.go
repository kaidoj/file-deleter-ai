package ai

// FeedForward calculates prediction
func FeedForward(m *Model, ctx *Context) *Context {
	ctx.hiddenInputs = Dot(m.Inputs, m.weights.T())
	ctx.hiddenPredictions = Apply(calcSigmoid, ctx.hiddenInputs)
	ctx.outputs = Dot(ctx.hiddenPredictions, m.outputWeights.T())
	ctx.outputPredictions = Apply(calcSigmoid, ctx.outputs)
	ctx.outputErrors = Substract(m.Outputs, ctx.outputPredictions)
	ctx.hiddenErrors = Dot(ctx.outputErrors, m.outputWeights)
	return ctx
}

func calcSigmoid(_, _ int, v float64) float64 {
	return Sigmoid(v)
}
