package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/kaidoj/file-deleter-ai/ai"
)

func main() {
	sleepTime2 := 2 * time.Second
	time.Sleep(sleepTime2)
	fmt.Println("Tere! Mina olen Juku.")
	time.Sleep(sleepTime2)
	fmt.Println("Kohe uurin kas sul on faile, mis on vanemad kui kolm kuud ja oleks vaja kustutada.")
	time.Sleep(sleepTime2)
	fmt.Println("Kontrollin... Palun oota")
	runAi()
	time.Sleep(sleepTime2)
	fmt.Println("Kui eksisin mõnega, siis ära pane pahaks. Ma olen veel üpris nooreke ;)")
	time.Sleep(sleepTime2)

	for {

	}
}

func runAi() {
	in, files := getDataFromFiles()
	weights, outputWeights := ai.Load("./models/hweights.model", "./models/oweights.model")
	predictions := ai.Predict(weights, outputWeights, in)

	del := 0
	for i := 0; i < len(files)-1; i++ {
		p := predictions.At(i, 0)
		if math.Round(p) < 1 {
			continue
		}

		percent := math.Round((p / 1) * 100)
		fmt.Printf("Mina arvan %v%%, et %v tuleb kustutada\r\n", percent, files[i])
		del++
	}

	if del != 0 {
		fmt.Printf("Ledisin, et kustutamist tahab %v faili\r\n", del)
	} else {
		fmt.Println("Ühtegi faili pole vaja kustutada :)")
	}
}

func getDataFromFiles() (*mat.Dense, []string) {
	return searchFiles(".")
}

// search files from directory
// calculate month for files
// return matrix and filelist
func searchFiles(dir string) (*mat.Dense, []string) {

	var response []float64
	var filelist []string
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		month := monthsCountSince(file.ModTime())
		response = append(response, float64(month))
		filelist = append(filelist, file.Name())
	}

	return mat.NewDense(len(response), 1, response), filelist
}

// monthsCountSince calculates the months between now
// and the createdAtTime time.Time value passed
func monthsCountSince(createdAtTime time.Time) int {
	now := time.Now()
	months := 0
	month := createdAtTime.Month()
	for createdAtTime.Before(now) {
		createdAtTime = createdAtTime.Add(time.Hour * 24)
		nextMonth := createdAtTime.Month()
		if nextMonth != month {
			months++
		}
		month = nextMonth
	}

	return months
}
