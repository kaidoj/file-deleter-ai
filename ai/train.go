package ai

import (
	"gonum.org/v1/gonum/mat"
)

// Context holds some data we pass down between functions
type Context struct {
	inputs      mat.Matrix
	targets     *mat.Dense
	errors      []*mat.Dense
	predictions []*mat.Matrix
}

// Train the model
func Train(model *Model, inputs, targets *mat.Dense) {

	ctx := &Context{}
	ctx.targets = targets

	for i := 0; i <= model.epochs; i++ {
		//get all data rows
		r, _ := inputs.Dims()
		for j := 1; j <= r; j++ {
			inp := inputs.Slice(j-1, j, 0, model.inputs)
			ctx.inputs = inp
			ctx = FeedForward(model, ctx)
		}

		for e := 0; e < len(ctx.errors); e++ {
			MatPrint(ctx.errors[e])
		}

		Backpropagation(model, ctx)
	}
}
