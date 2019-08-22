package ai

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func ConfusionMatrix(p, in, out mat.Matrix) {

}

func sums(m mat.Matrix) []float64 {
	var uniques []float64
	r, _ := m.Dims()
	for i := 0; i < r; i++ {
		v := math.Round(m.At(i, 0))

		uniques = append(uniques, v)
	}

	return uniques
}

func uniques(m mat.Matrix) []float64 {
	var uniques []float64
	r, _ := m.Dims()
	for i := 0; i < r; i++ {
		v := m.At(i, 0)
		ok, _ := InArray(v, uniques)
		if ok {
			continue
		}

		uniques = append(uniques, v)
	}

	return uniques
}
