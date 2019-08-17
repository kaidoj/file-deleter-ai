package ai

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"os"

	"gonum.org/v1/gonum/mat"
)

type DataReader interface {
	Read(filename string, iCols, oCols []int) (*mat.Dense, *mat.Dense)
}

// Read loads CSV file contents into Matrix
func Read(filename string, iCols, oCols []int) (*mat.Dense, *mat.Dense) {
	csvFile, _ := os.Open(filename)
	reader := csv.NewReader(bufio.NewReader(csvFile))
	reader.Comma = ';'

	var inp []float64
	var outp []float64

	n := 0
	for {

		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatalf("Parsin CSV failed %v", err)
		}

		if n == 0 {
			n++
			continue
		}

		for i, v := range line {
			ok, _ := InArray(i, iCols)
			if ok {
				f := String2float64(v)
				inp = append(inp, f)
			}

			ok2, _ := InArray(i, oCols)
			if ok2 {
				f := String2float64(v)
				outp = append(outp, f)
			}
		}
		n++
	}

	return mat.NewDense(n-1, len(iCols), inp), mat.NewDense(n-1, len(oCols), outp)
}
