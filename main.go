package main

import (
	"github.com/kaidoj/file-deleter-ai/ai"
	"gonum.org/v1/gonum/mat"
)

func main() {

	//test()
	//os.Exit(1)

	// define input cols in data
	iCols := []int{2, 3}
	// define output cols in data
	oCols := []int{3}
	outp := []float64{1}
	// number of inputs and outputs
	nrOfInputs := len(iCols)
	nrOfOutputs := len(outp)

	inp, _ := ai.Read("data/train_keep.csv", iCols, oCols)
	//ai.MatPrint(inp)
	//ai.MatPrint(outp)

	// calculate hidden layers
	// formula: ((inputs+outputs) * 2) / nr smaples
	// ((2 + 2) * 2)) / 3
	//hiddenLayers := []int{3}
	//TODO: calc more then one hidden layer
	nrOfNodes := ai.NrOfNodes(nrOfInputs, nrOfOutputs, inp)
	layers := []int{nrOfNodes, len(outp)}

	model := ai.NewModel(layers, nrOfInputs, nrOfOutputs, 0.1, 2)
	ai.Train(model, inp, mat.NewDense(nrOfOutputs, 1, outp))
}

func test() {

	var inputs = []float64{1, 0}
	i := mat.NewDense(2, 1, inputs)
	ai.MatPrint(i)

	var weights = []float64{2, 2}
	w := mat.NewDense(2, 1, weights)
	ai.MatPrint(w)

	m := new(mat.Dense)
	m.Sub(i, w)

	ai.MatPrint(m)

}
