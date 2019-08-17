package ai

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Cost(prediction float64, target float64) float64 {
	return (prediction - target) * 2
}

// NrOfNodes calculates number of nodes for hidden layer
func NrOfNodes(inputs, outputs int, data *mat.Dense) int {
	i := float64(inputs)
	o := float64(outputs)
	r, _ := data.Dims()
	return int(math.RoundToEven(((i + o) * 2) / float64(r)))
}
