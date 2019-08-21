package ai

import (
	"fmt"
	"log"
	"reflect"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// InArray checks if value is in give array
func InArray(v interface{}, in interface{}) (ok bool, i int) {
	val := reflect.Indirect(reflect.ValueOf(in))
	switch val.Kind() {
	case reflect.Slice, reflect.Array:
		for ; i < val.Len(); i++ {
			if ok = v == val.Index(i).Interface(); ok {
				return
			}
		}
	}
	return
}

// String2float64 coverts string to float64
func String2float64(v string) float64 {
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		log.Fatalf("Error parsing field as float. %v", err)
	}

	return f
}

// MatPrint prints matrix data readable format
func MatPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// ToString converts float64 to string
func ToString(floats []float64) []string {
	var strings []string
	for _, f := range floats {
		strings = append(strings, strconv.FormatFloat(f, 'f', -1, 64))
	}

	return strings
}
