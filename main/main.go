package main

import (
	"fmt"
	"func_master"
	"math"
	"sort"
	"strings"
)

var (
	worker func_master.Worker
	m      = make(map[int]int)
)

const (
	machi = int64(10)
)

func main() {
	var pic func_master.PictureNodeInterface

	node := &func_master.ItemGraph{}
	pic = node
	for i := 1; i <= 9; i++ {
		pic.AddNode(&func_master.PictureNode{i})
	}
	//生成边
	A := []int{1, 1, 2, 2, 2, 3, 4, 5, 5, 6, 1}
	B := []int{2, 5, 3, 4, 5, 4, 5, 6, 7, 8, 9}
	for i := 0; i < 11; i++ {
		pic.AddEdges(&func_master.PictureNode{A[i]}, &func_master.PictureNode{B[i]})
	}
	pic.String()

}

func readBinaryWatch(turnedOn int) []string {
	result := []string{}
	for i := 0; i < 12; i++ {
		for j := 0; j < 60; j++ {
			b1 := fmt.Sprintf("%b", i)
			b2 := fmt.Sprintf("%b", j)
			sumOne := strings.Count(b1, "1") + strings.Count(b2, "1")
			fmt.Println(b1, b2)
			if sumOne == turnedOn {
				result = append(result, fmt.Sprintf("%d:%02d", i, j))
			}
		}
	}
	return result

}

func checkOnesSegment(s string) bool {
	//if s =="1" {
	//	return true
	//}
	//var start uint8 = 49
	//num := 0
	return !strings.Contains(s, "01")
	//for i := 1; i < len(s); i++ {
	//	fmt.Println(len(s))
	//	num +=1
	//	if start == s[i] {
	//		return true
	//	}else if num<= len(s){
	//		start = s[i]
	//		i++
	//	}
	//}
	//return false
}

func getMoneyAmount(n int) int {
	// 采用动态规划实现
	money := make([][]int, n+1)
	// 先将数据补充
	for i := 0; i <= n; i++ {
		money[i] = make([]int, n+1)
	}
	for i := n; i >= 1; i-- {
		for j := i; j <= n; j++ {
			if i == j {
				money[i][j] = 0
			} else {
				money[i][j] = (1 << 31) - 1
				for x := i; i < j; x++ {
					money[i][j] = min(money[i][j], max(money[i][x-1], money[x+1][j])+x)
				}
			}

		}
	}
	return money[1][n]
}

func min(a, b int) int {
	if a > b {
		return b
	} else {
		return a
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mySqrt(x int) int {
	middle := 1
	start := 0
	for {
		start = middle * middle
		if start == x {
			return middle
		}
		if start < x {
			middle = middle * 2
			continue
		}

		return binary(middle/2, middle, x)
	}
}

func binary(start, end, x int) int {
	sum := 0
	for start <= end {
		mid := (start + end) / 2
		sum = mid * mid
		if sum == x {
			return mid
		} else if sum < x {
			start = mid + 1
		} else {
			end = mid - 1
		}

	}
	return end
}

func coinChange(coins []int, amount int) int {
	// 背包问题·
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
		for _, coin := range coins {
			if i >= coin && dp[i-coin]+1 < dp[i] {
				fmt.Println(i, coin, dp[i-coin]+1, dp[i])
				dp[i] = dp[i-coin] + 1
			}
		}
	}
	if dp[amount] == amount+1 {
		return -1
	}
	return dp[amount]
}
func coinChange1(coins []int, amount int) int {
	count := make([]int, amount+1)
	for i := 1; i < len(count); i++ {
		count[i] = math.MaxInt32
		for _, j := range coins {
			if i >= j {
				count[i] = min(count[i], count[i-j]+1)
			}
		}
	}
	if count[amount] > amount {
		return -1
	} else {
		return count[amount]
	}
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

// value: 硬币币值, n: 硬币数量, w: 支付金额
func lfchange(value []int, n int, w int) int {
	sort.Ints(value)
	// 最小币值
	minV := value[0]
	// dp[i]表示支付金额为i需要多少个硬币
	dp := make([]int, w+1)
	for _, v := range value { // 初始化状态
		if v > w {
			break
		}
		dp[v] = 1
	}
	// 硬币数
	var count int
	for i := minV + 1; i <= w; i++ { // 动态规划方程转移
		count = 0
		for j := n - 1; j >= 0; j-- {
			if i%value[j] == 0 {
				dp[i] = i / value[j]
				break
			}
		}
		for j := minV; j < i; j++ {
			if dp[j] != 0 && dp[i-j] != 0 {
				count = dp[j] + dp[i-j]
				if count < dp[i] {
					dp[i] = count
				}
			}
		}
	}
	return dp[w]
}
