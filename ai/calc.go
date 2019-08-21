package ai

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func SigmoidPrime(m mat.Matrix) *mat.Dense {
	rows, cols := m.Dims()
	o := make([]float64, rows*cols)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, cols, o)
	return MultiplyElem(m, Substract(ones, m))
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

func Multiply(m, n mat.Matrix) *mat.Dense {
	r := new(mat.Dense)
	r.Mul(m, n)
	return r
}

func MultiplyElem(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func Substract(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func Dot(m, n mat.Matrix) *mat.Dense {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func Scale(f float64, m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	s := mat.NewDense(r, c, nil)
	s.Scale(f, m)
	return s
}

func Add(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}
