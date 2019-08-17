package main

import (
	"encoding/csv"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
)

// Generate dummy data
func main() {

	rand.Seed(time.Now().Unix())
	var data [][]string

	data = append(data, []string{
		"file",
		"created_at",
		"months",
		"delete",
	})

	data = generateRows(data)

	file, err := os.Create("train.csv")
	checkError("Cannot create file", err)
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Comma = ';'
	defer writer.Flush()

	for _, value := range data {
		err := writer.Write(value)
		checkError("Cannot write to file", err)
	}
}

func checkError(message string, err error) {
	if err != nil {
		log.Fatal(message, err)
	}
}

func generateRows(data [][]string) [][]string {

	time := time.Now()
	for i := 1; i <= 100; i++ {
		rnd := randNr(1, 5)
		ntime := time.AddDate(0, -rnd, 0)

		del := "1"
		if rnd < 3 {
			del = "0"
		}

		r := strconv.FormatInt(int64(rnd), 16)

		filename := "file" + r + ".jpg"
		data = append(data, []string{
			filename,
			ntime.Format("2006-01-02 15:04:05"),
			r,
			del,
		})
	}

	return data
}

func randNr(a, b int) int {
	n := rand.Intn((b - a) + a)
	return n
}
