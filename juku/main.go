package main

import (
	"github.com/kaidoj/file-deleter-ai/ai"
)

func main() {
	// define input cols in data
	iCols := []int{2} //3
	// define output cols in data
	oCols := []int{3}
	in, out := ai.Read("../data/train.csv", iCols, oCols)
	weights, outputWeights := ai.Load("../models/hweights.model", "../models/oweights.model")
	_, predictions := ai.Predict(weights, outputWeights, in, out)
	ai.Accuracy(predictions, in, out)

	/*sleepTime2 := 2 * time.Second
	sleepTime3 := 3 * time.Second
	time.Sleep(sleepTime3)
	fmt.Println("Tere! Mina olen Juku.")
	time.Sleep(sleepTime2)
	fmt.Println("Kohe uurin kas sul on faile, mis on vanemad kui kolm kuud ja oleks vaja kustutada.")
	time.Sleep(sleepTime2)
	fmt.Println("Kontrollin... Palun oota")

	time.Sleep(sleepTime3)
	fmt.Println("Kui eksisin mõnega, siis ära pane pahaks. Ma olen veel üpris nooreke ;)")
	time.Sleep(sleepTime3)

	for {
		go func() {
			time.Sleep(sleepTime2)
		}()
	}*/
}
