package ai

import (
	"fmt"
	"math"
)

func Accuracy(m *Model, ctx *Context) float64 {

	oe := Apply(calcAbs, ctx.OutputErrors)
	r, _ := oe.Dims()
	for i := 0; i < r; i++ {
		output := m.Outputs.At(i, 0)
		fmt.Println(output)
		got := oe.At(i, 0)
		percent := 100 - math.Round((got/1)*100)

		fmt.Printf("Out %v; Err %v; Perc %v% \r\n", output, got, percent)

	}

	return 00
}

func calcAbs(_, _ int, v float64) float64 {
	return math.Abs(v)
}
