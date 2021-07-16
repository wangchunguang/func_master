package main

import (
	"fmt"
	"func_master"
)

var (
	worker func_master.Worker
	m      = make(map[int]int)
)

const (
	machi = int64(10)
)

func main() {
	arr := []int{23, 13, 34, 32, 65, 43, 76, 34, 2, 4, 26, 15}
	func_master.QuickSort(arr, 0, len(arr)-1)
	fmt.Println(arr)
	//demo8(arr, 0, len(arr)-1)
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

func demo2(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 2
	}
	if n == 3 {
		return 3
	}
	if value, ok := m[n]; ok {
		return value
	}
	num := demo2(n-1) + demo2(n-2)
	m[n] = num
	return num
}

func demo3(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 2
	}
	if n == 3 {
		return 3
	}

	num1 := 0 // 当前
	num2 := 2 // 上一位
	num3 := 1 // 上上一位

	for i := 3; i <= n; i++ {
		num1 = num2 + num3
		num3 = num2
		num2 = num1
	}
	return num1
}

func demo4(arr []int) {
	count := len(arr)
	if count <= 1 {
		return
	}
	for i := 1; i < count; i++ {
		// 假设左边都是有序的
		value := arr[i]
		k := i - 1
		// 每次减1 并且左边的数据比右边的数据大 就会开始进行计算
		for k >= 0 && arr[k] > value {
			arr[k+1] = arr[k]
			k -= 1
		}
		arr[k+1] = value
	}
	fmt.Println(arr)
}

func demo5(arr []int) {
	count := len(arr)
	if count <= 0 {
		return
	}
	for i := 0; i < count; i++ {
		for k := 0; k < count-i-1; k++ {
			if arr[k] > arr[k+1] {
				arr[k], arr[k+1] = arr[k+1], arr[k]
			}
		}
	}
	fmt.Println(arr)
}

func demo6(arr []int) []int {
	count := len(arr)
	if count <= 1 {
		return arr
	}
	mid := count / 2
	left := demo6(arr[:mid])
	right := demo6(arr[mid:])
	return demo7(left, right)
}

func demo7(left, right []int) []int {
	arr := []int{}
	//	 因为两个都是有序的  所以直接进行对比，左边和右边的比较
	l, r := 0, 0
	for l < len(left) && r < len(right) {
		if left[l] < right[r] {
			arr = append(arr, left[l])
			l += 1
		} else {
			arr = append(arr, right[r])
			r += 1
		}
	}
	arr = append(arr, left[l:]...)
	arr = append(arr, right[r:]...)
	return arr
}

func demo8(arr []int, left, right int) {
	if left < right {
		p := demo9(arr, left, right)
		demo8(arr, left, p-1)
		demo8(arr, p+1, right)
	}
}

func demo9(arr []int, left, right int) int {
	fmt.Println(arr)
	// 导致第一个位置的数据为空
	value := arr[left]
	for left < right {
		for left < right && value <= arr[right] {
			right--
		}
		arr[left] = arr[right]

		for left < right && arr[left] <= value {
			left++
		}
		arr[right] = arr[left]
	}
	arr[left] = value
	fmt.Println(arr)
	return left
}
