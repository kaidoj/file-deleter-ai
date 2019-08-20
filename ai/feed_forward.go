package ai

import "fmt"

// FeedForward calculates prediction
func FeedForward(m *Model, ctx *Context) *Context {

	MatPrint(m.weights)
	fmt.Println("weights")
	MatPrint(m.Inputs)
	fmt.Println("inputs")
	ctx.hiddenInputs = Dot(m.weights, m.Inputs)
	MatPrint(ctx.hiddenInputs)
	fmt.Println("h inputs")
	ctx.hiddenPredictions = Apply(calcSigmoid, ctx.hiddenInputs)
	MatPrint(ctx.hiddenPredictions)
	fmt.Println("predictions")
	MatPrint(m.outputWeights)
	fmt.Println("o weights")
	ctx.outputs = Dot(m.outputWeights, ctx.hiddenPredictions)
	MatPrint(ctx.outputs)
	fmt.Println("o outputs")
	ctx.outputPredictions = Apply(calcSigmoid, ctx.outputs)
	MatPrint(ctx.outputPredictions)
	fmt.Println("o predictions")
	MatPrint(m.Outputs)
	fmt.Println("targets")
	ctx.outputErrors = Substract(m.Outputs, ctx.outputPredictions)
	MatPrint(ctx.outputErrors)
	fmt.Println("outputErrors")
	ctx.hiddenErrors = Dot(m.outputWeights.T(), ctx.outputErrors)
	MatPrint(ctx.hiddenErrors)
	fmt.Println("hiddenErrors")
	return ctx
}

func calcSigmoid(_, _ int, v float64) float64 {
	return Sigmoid(v)
}
