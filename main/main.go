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

	list := func_master.List{}
	fmt.Println(list.IsEmpty())
	list.HeadSave(1)
	list.HeadSave(2)
	list.HeadSave(3)
	list.HeadSave(4)
	list.HeadSave(5)
	list.TailSave("a")
	list.TailSave(4)
	list.TailSave("c")
	list.TailSave(4)
	list.DeleteAtValue(4)
	list.RangeList()

	//c := make(chan int, 1)
	//go demo1(c)
	//sig := make(chan os.Signal, 1)
	//signal.Notify(sig)
	//select {
	//case <-sig:
	//
	//}
	//fmt.Println(1111111111)
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
