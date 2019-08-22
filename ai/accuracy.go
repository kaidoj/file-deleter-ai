package ai

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Accuracy(p, in, out mat.Matrix) (int, int) {
	tp := getTruePositives(p, out)
	fmt.Printf("True positives: %v\r\n", tp)
	a := getAccuracy(tp, out)
	fmt.Printf("Accuracy %v percent\r\n", a)

	return tp, a
}

func getTruePositives(m, out mat.Matrix) int {
	res := 0
	r, _ := m.Dims()
	for i := 0; i < r; i++ {
		v := m.At(i, 0)
		actual := out.At(i, 0)
		//fmt.Printf("Pred: %v; Actual: %v\r\n", math.Round(v), actual)
		if math.Round(v) != actual {
			continue
		}

		res++
	}

	return res
}

func getAccuracy(tp int, out mat.Matrix) int {
	r, _ := out.Dims()

	a := (tp / r) * 100

	return a
}
