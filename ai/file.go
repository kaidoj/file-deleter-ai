package ai

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

// Save weights to file
func Save(weights, outputWeights *mat.Dense) {
	h, err := os.Create("models/hweights.model")
	defer h.Close()
	if err == nil {
		weights.MarshalBinaryTo(h)
	}
	o, err := os.Create("models/oweights.model")
	defer o.Close()
	if err == nil {
		outputWeights.MarshalBinaryTo(o)
	}
}

// load weights from file
func Load(weightsFile, oWeightsFile string) (*mat.Dense, *mat.Dense) {
	weights := new(mat.Dense)
	oWeights := new(mat.Dense)
	h, err := os.Open(weightsFile)
	defer h.Close()
	if err == nil {
		weights.UnmarshalBinaryFrom(h)
	} else {
		fmt.Println(err)
	}
	o, err := os.Open(oWeightsFile)
	defer o.Close()
	if err == nil {
		oWeights.UnmarshalBinaryFrom(o)
	} else {
		fmt.Println(err)
	}

	return weights, oWeights
}
