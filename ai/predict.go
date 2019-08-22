package ai

import (
	"gonum.org/v1/gonum/mat"
)

func Predict(weights, outputWeights, inputs, outputs mat.Matrix) (mat.Matrix, mat.Matrix) {
	hiddenInputs := Dot(inputs, weights.T())
	hiddenPredictions := Apply(calcSigmoid, hiddenInputs)
	outputsDot := Dot(hiddenPredictions, outputWeights.T())
	outputPredictions := Apply(calcSigmoid, outputsDot)
	errors := Substract(outputs, outputPredictions)
	return errors, outputPredictions
}
