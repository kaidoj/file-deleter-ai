package ai

import (
	"gonum.org/v1/gonum/mat"
)

// Context holds some data we pass down between functions
type Context struct {
	hiddenInputs      mat.Matrix
	hiddenPredictions mat.Matrix
	outputs           mat.Matrix
	outputPredictions mat.Matrix
	OutputErrors      mat.Matrix
	hiddenErrors      mat.Matrix
}

// Train the model
func Train(model *Model) (*Model, *Context, mat.Matrix, mat.Matrix) {
	ctx := &Context{}
	for i := 0; i <= model.Epochs; i++ {
		FeedForward(model, ctx)
		Backpropagation(model, ctx)
	}

	//fmt.Println("Training results")
	//MatPrint(ctx.OutputErrors)
	//fmt.Println(Accuracy(model, ctx))

	return model, ctx, model.weights, model.outputWeights
}
