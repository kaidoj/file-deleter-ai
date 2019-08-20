package ai

import "fmt"

// Backpropagation recalculates weights and biases
func Backpropagation(m *Model, ctx *Context) {

	//output layer
	MatPrint(ctx.outputErrors)
	fmt.Println("outputErrors")

	MatPrint(ctx.outputPredictions)
	fmt.Println("o predictions")

	MatPrint(ctx.hiddenPredictions)
	fmt.Println("h predictions")

	MatPrint(m.outputWeights)
	fmt.Println("output weights")

	mul := Multiply(SigmoidPrime(ctx.outputPredictions), ctx.outputErrors.T())
	MatPrint(mul)
	fmt.Println("multiply")

	d := Dot(mul, ctx.hiddenPredictions)
	MatPrint(d)
	fmt.Println("dot")

	s := Scale(m.LearingRate, d)
	MatPrint(s)
	fmt.Println("scale")

	a := Add(m.outputWeights, s)
	MatPrint(a)
	fmt.Println("add")

	m.weights = a

	// hidden layer
	/*m.weights = Add(m.weights,
	Scale(m.LearingRate,
		Dot(Multiply(ctx.hiddenErrors, SigmoidPrime(ctx.hiddenPredictions)),
			ctx.inputs)))*/
}
