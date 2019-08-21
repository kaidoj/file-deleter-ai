package ai

// Backpropagation recalculates weights and biases
func Backpropagation(m *Model, ctx *Context) {

	//output layer
	mul := MultiplyElem(ctx.OutputErrors, SigmoidPrime(ctx.outputPredictions))
	d := Multiply(mul.T(), ctx.hiddenPredictions)
	s := Scale(m.LearingRate, d)
	a := Add(s, m.outputWeights)
	m.outputWeights = a

	//hidden layer
	hmul := MultiplyElem(ctx.hiddenErrors, SigmoidPrime(ctx.hiddenPredictions))
	hd := Multiply(hmul.T(), m.Inputs)
	hs := Scale(m.LearingRate, hd)
	ha := Add(hs, m.weights)
	m.weights = ha
}
