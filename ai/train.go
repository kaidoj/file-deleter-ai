package ai

import (
	"gonum.org/v1/gonum/mat"
)

func Train(model *Model, inputs, targets *mat.Dense) {

	//for i := 0; i <= model.epochs; i++ {
	//get all data rows
	r, _ := inputs.Dims()
	for j := 1; j <= r; j++ {
		inp := inputs.Slice(j-1, j, 0, model.inputs)
		errors := FeedForward(model, inp, targets)
		MatPrint(errors)
	}
	//}
}
