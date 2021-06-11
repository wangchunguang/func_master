package main

import (
	"fmt"
	"func_master"
)

var worker func_master.Worker

const (
	machi = int64(4)
)

func main() {
	num := func_master.NewWorker(int64(machi))
	fmt.Println(num.GetId())
}
