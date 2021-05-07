package main

import (
	"fmt"
	"time"
)

var c = make(chan int, 1)

func d1(cc chan int) {
	for i := 0; i < 2; i++ {

	}
	time.Sleep(5 * time.Second)
	cc <- 10
}

func main() {
	go d1(c)

	<-c
	fmt.Println(222222222)
}
