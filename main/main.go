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
	queue := func_master.NewArrayQueue()

	queue.Insert(1)
	queue.Insert(2)
	queue.Insert(3)
	queue.Insert(4)
	queue.Insert(5)

	fmt.Println(queue)
	fmt.Println(queue.DeQueue())
	fmt.Println(queue.DeQueue())
	fmt.Println(queue.DeQueue())
	fmt.Println(queue.DeQueue())
	fmt.Println(queue.DeQueue())
	fmt.Println(queue.DeQueue())
	fmt.Println(queue)

	//list := func_master.List{}
	//fmt.Println(list.IsEmpty())
	//list.HeadSave(1)
	//list.HeadSave(2)
	//list.HeadSave(3)
	//list.HeadSave(4)
	//list.HeadSave(5)
	//list.HeadSave(6)
	//list.HeadSave(7)
	//list.HeadSave(8)
	//list.HeadSave(9)
	//list.RangeList()
	//reverseList := list.ReverseList()
	//reverseList.RangeList()
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
