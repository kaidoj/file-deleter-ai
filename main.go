package main

import (
	"fmt"

	"github.com/kaidoj/file-deleter-ai/ai"
)

func main() {

	// define input cols in data
	iCols := []int{2} //3
	// define output cols in data
	oCols := []int{3}
	// define hidden layers and neurons
	hiddenLayerNeurons := 5 // 3

	// setup data
	inp, outp := ai.Read("data/train500.csv", iCols, oCols)

	config := &ai.ModelConfig{
		0.001,
		20000,
		hiddenLayerNeurons,
		len(iCols),
		len(oCols),
		inp,
		outp,
	}

	model := ai.NewModel(config)
	m, ctx := ai.Train(model)
	// setup data
	in, out := ai.Read("data/train_keep1.csv", iCols, oCols)
	m.Inputs = in
	m.Outputs = out
	errors := ai.Predict(m, ctx)
	fmt.Println("Test results")
	ai.MatPrint(errors)
}
