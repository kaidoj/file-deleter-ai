package ai

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Context holds some data we pass down between functions
type Context struct {
	hiddenInputs      mat.Matrix
	hiddenPredictions mat.Matrix
	outputs           mat.Matrix
	outputPredictions mat.Matrix
	outputErrors      mat.Matrix
	hiddenErrors      mat.Matrix
}

// Train the model
func Train(model *Model) {
	ctx := &Context{}
	for i := 0; i <= model.Epochs; i++ {
		FeedForward(model, ctx)
		Backpropagation(model, ctx)
	}

	fmt.Println("Training results")
	MatPrint(ctx.outputErrors)
}
