package main

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"
)

type TestStruct struct{}

func NilOrNot(v interface{}) bool {
	fmt.Println(v)
	return v == nil
}

func (t *TestStruct) Name(int2 int) (i, j int) {
	fmt.Println(int2)
	fmt.Println("111111")
	return 9, 9
}

func main() {

	http.ListenAndServe(":8080", http.FileServer(http.Dir("/files/path")))
}

func demo(arr []int, sy *sync.WaitGroup, k, v int) {
	defer sy.Done()
	arr[k] = v
}

func handle(ctx context.Context, duration time.Duration) {
	select {
	case <-ctx.Done():
		fmt.Println("handle", ctx.Err())
	case <-time.After(duration):
		fmt.Println("process request with", duration)
	}
}
