package main

import (
	"github.com/kaidoj/file-deleter-ai/ai"
)

func main() {

	// define input cols in data
	iCols := []int{2, 3}
	// define output cols in data
	oCols := []int{3}
	// define hidden layers and neurons
	hiddenLayerNeurons := 3

	// setup data
	inp, outp := ai.Read("data/train_keep1.csv", iCols, oCols)

	config := &ai.ModelConfig{
		0.1,
		2,
		hiddenLayerNeurons,
		len(iCols),
		len(oCols),
		inp,
		outp,
	}

	model := ai.NewModel(config)
	ai.Train(model)
}
