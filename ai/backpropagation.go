package ai

import (
	"fmt"
)

// Backpropagation recalculates weights and biases
func Backpropagation(model *Model, ctx *Context) {
	//fmt.Println("err")
	maxW := len(ctx.weights) - 1
	for i := maxW; i >= 0; i-- {
		// output layer
		if i == maxW {
			m := Multiply(SigmoidPrime(ctx.predictions[i]), ctx.errors)
			dt := Dot(ctx.predictions[i-1], m)
			MatPrint(dt)
			fmt.Println("gradient")
			s := Scale(model.learingRate, dt)
			MatPrint(s)
			fmt.Println("s")
			MatPrint(ctx.weights[i])
			d := Add(s, ctx.weights[i].T())
			fmt.Println("old")
			ctx.weights[i] = d
			MatPrint(ctx.weights[i])
			fmt.Println("new")

		} else {
			m := Multiply(SigmoidPrime(ctx.predictions[i]), ctx.errors)
			MatPrint(m)
			fmt.Println("weight b")
			MatPrint(ctx.inputs.T())
			fmt.Println("inputs")
			dt := Multiply(m, ctx.inputs)
			MatPrint(dt)
			fmt.Println("gradient")
			s := Scale(model.learingRate, dt)
			MatPrint(s)
			fmt.Println("s")
			MatPrint(ctx.weights[i])
			fmt.Println("rweights")
			d := Add(s, ctx.weights[i])
			MatPrint(d)
			fmt.Println("d")
			ctx.weights[i] = d
			MatPrint(model.weights[i])
			fmt.Println("new")
		}
	}
}
