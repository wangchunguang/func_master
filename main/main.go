package main

import (
	"fmt"
	"func_master"
	"os"
	"os/signal"
)

var worker func_master.Worker

const (
	machi = int64(4)
)

func main() {
	c := make(chan int, 1)
	go demo1(c)
	sig := make(chan os.Signal, 1)
	signal.Notify(sig)
	select {
	case <-sig:

	}
	fmt.Println(1111111111)
}

func demo1(c chan int) {
	c <- 2
	func_master.Go(func() {
		select {
		case num := <-c:
			fmt.Println(num)
		default:
			fmt.Println(555555555555)
		}
	})

}

func demo2(c chan int) {

}
