package ai

import (
	"gonum.org/v1/gonum/mat"
)

// Context holds some data we pass down between functions
type Context struct {
	inputs      mat.Matrix
	target      mat.Matrix
	errors      *mat.Dense
	weights     []mat.Matrix
	predictions []mat.Matrix
	prediction  mat.Matrix
}

// Train the model
func Train(model *Model, inputs, targets *mat.Dense) {

	ctx := &Context{}
	ctx.weights = model.weights

	for i := 0; i <= model.epochs; i++ {
		//fmt.Printf("---epoch %v---\r\n", i)
		//get all data rows
		r, _ := inputs.Dims()
		for j := 1; j <= r; j++ {
			ctx.inputs = inputs.Slice(j-1, j, 0, model.inputs)
			ctx.target = targets.Slice(j-1, j, 0, model.outputs)
			ctx = FeedForward(model, ctx)
			//fmt.Printf("Predicted output: %v and Actual: %v \r\n", ctx.prediction, ctx.target)
			//os.Exit(1)
			Backpropagation(model, ctx)
		}
		//fmt.Println("Final error")
		MatPrint(ctx.errors)
	}

	//fmt.Println("Final error")
	//MatPrint(ctx.errors)
}
