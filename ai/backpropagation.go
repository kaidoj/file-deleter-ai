package ai

// Backpropagation recalculates weights and biases
func Backpropagation(model *Model, ctx *Context) {

	/*d := new(mat.Dense)
	d.Sub(ctx.errors, ctx.targets)
	deltas := new(mat.Dense)
	deltas.Mul(ctx.errors, d)*/

	/*var newWeights []*mat.Dense
	for i := 0; i < len(model.weights); i++ {
		newWeight := new(mat.Dense)
		newWeight.Scale(model.learingRate, model.weights[i])
		newWeights = append(newWeights, newWeight)
	}

	model.weights = newWeights*/

}
