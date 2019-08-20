package main

import (
	"fmt"

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
	//outp := []float64{1}
	// number of inputs and outputs
	nrOfInputs := len(iCols)
	nrOfOutputs := 1

	inp, outp := ai.Read("data/train_keep1.csv", iCols, oCols)
	//ai.MatPrint(inp)
	//ai.MatPrint(outp)

	// calculate hidden layers
	// formula: ((inputs+outputs) * 2) / nr smaples
	// ((2 + 2) * 2)) / 3
	//hiddenLayers := []int{3}
	//TODO: calc more then one hidden layer
	//nrOfNodes := ai.NrOfNodes(nrOfInputs, nrOfOutputs, inp)
	//fmt.Println(nrOfNodes)
	layers := []int{3, 1}
	model := ai.NewModel(layers, nrOfInputs, nrOfOutputs, 0.1, 2)
	ai.Train(model, inp, outp)
}

func test() {

	var inputs = []float64{1, 2}
	i := mat.NewDense(1, 2, inputs)
	ai.MatPrint(i)
	fmt.Println("inputs")

	var predictions = []float64{1, 2}
	p := mat.NewDense(2, 1, predictions)
	ai.MatPrint(p)
	fmt.Println("predictions")

	var errors = []float64{2, 2}
	e := mat.NewDense(1, 2, errors)
	ai.MatPrint(e)
	fmt.Println("errors")

	var weights = []float64{1, 2, 3, 4}
	w := mat.NewDense(2, 2, weights)
	ai.MatPrint(w)
	fmt.Println("weights")

	m := new(mat.Dense)
	//m.Mul(i, w)
	//m.Mul(e, p)
	m.Product(e, p)

	ai.MatPrint(m)

}
