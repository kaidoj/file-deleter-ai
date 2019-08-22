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
		0.0002,
		35000,
		hiddenLayerNeurons,
		len(iCols),
		len(oCols),
		inp,
		outp,
	}

	model := ai.NewModel(config)
	_, _, weights, outputWeights := ai.Train(model)
	ai.Save(weights, outputWeights)
	// setup data
	in, out := ai.Read("data/train_keep_large.csv", iCols, oCols)
	//weights, outputWeights := ai.Load()
	errors, predictions := ai.Predict(weights, outputWeights, in, out)
	fmt.Println("Test results")
	ai.MatPrint(predictions)
	fmt.Println("predictions")
	ai.MatPrint(errors)
	fmt.Println("Accuracy")
	ai.Accuracy(predictions, in, out)
}
