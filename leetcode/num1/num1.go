package num1

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

func solveSudoku(board [][]byte) {
	// 记录某行
	var row [9][9]bool
	// 记录某列
	var col [9][9]bool
	// box记录
	var box [9][9]bool

	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			//	判断当前不是数字的时候
			if board[i][j] != '.' {
				// 初始化
				num := board[i][j] - '1'
				boxK := j/3 + (i/3)*3
				row[i][num], col[j][num], box[boxK][num] = true, true, true
			}

		}
	}
	//	 进行回溯算法
	fill(board, 0, row, col, box)
}

func fill(board [][]byte, n int, row, col, box [9][9]bool) bool {
	//	临界点  因为是9*9的二维数组
	if n == 81 {
		return true
	}
	// 每一行一行的运算
	rowK := n / 9
	colK := n % 9
	// 不表示有数字
	if board[rowK][colK] != '.' {
		return fill(board, n+1, row, col, box)
	}
	// 获取是第几个区间
	boxK := (rowK/3)*3 + (colK / 3)
	// 因为行，列，区间都是九个数
	for num := 0; num < 9; num++ {
		// 判断列 行 数组里面都不存在这个数
		if !row[rowK][num] && !col[colK][num] && !box[boxK][num] {
			board[rowK][colK] = byte('1' + num)
			row[rowK][num], col[colK][num], box[boxK][num] = true, true, true
			//	 下一个填充
			if fill(board, n+1, row, col, box) {
				return true
			}
			row[rowK][num], col[colK][num], box[boxK][num] = false, false, false //失败回溯
		}
	}
	board[rowK][colK] = '.'
	return false
}

func isValidSudoku(board [][]byte) bool {
	if len(board) == 0 {
		return false
	}
	// 记录某行
	var row [9][10]int
	// 记录某列
	var col [9][10]int
	// box记录
	var box [9][10]int

	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			//	判断当前不是数字的时候
			if board[i][j] == '.' {
				continue
			}
			//	获取数据
			curNumber := board[i][j] - '0'
			fmt.Println(curNumber)
			if row[i][curNumber] == 1 {
				return false
			}
			if col[j][curNumber] == 1 {
				return false
			}
			// 判断是第几个box，如果有重复的直接返回false
			if box[j/3+(i/3)*3][curNumber] == 1 {
				return false
			}

			row[i][curNumber] = 1
			col[j][curNumber] = 1
			box[j/3+(i/3)*3][curNumber] = 1
		}
	}
	return true
}

func searchInsert(nums []int, target int) int {
	if len(nums) == 0 {
		nums = append(nums, target)
		return 0
	}

	for i := 0; i < len(nums); i++ {
		if nums[i] == target {
			return i
		} else if nums[i] > target {
			return i
		}
	}
	return len(nums)
}

func searchRange(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}
	lenght := len(nums)
	left := 0
	right := sort.Search(lenght, func(i int) bool {
		return nums[i] > target
	})
	fmt.Println(left, right)
	return []int{left, right - 1}
}

func findMin(nums []int) int {
	sort.Ints(nums)
	return nums[0]
}

func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		fmt.Println(mid)
		if nums[mid] == target {
			return mid
		} else if nums[mid] >= nums[left] { // 判断左边是不是有序 如果判断错误，
			if nums[mid] > target && target >= nums[left] { // 判断目标值在 有序区间 左边内
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else if nums[mid] < nums[left] { // 判断左边不是有序，即有序在右
			if nums[mid] < target && target <= nums[right] { // 判断目标值在 有序区间 右边内
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1

}

// Bsearch 二分查找算法
func bsearch(arr []int, value int, left, right int) int {
	if left > right {
		return -1
	}
	mid := left + (right-left)/2
	if arr[mid] == value {
		return mid
	} else if arr[left] < value {
		return bsearch(arr, value, mid+1, right)
	} else {
		return bsearch(arr, value, left, mid-1)
	}

}

func longestValidParentheses(s string) int {
	// 栈底放最后一个未被匹配的右括号下标,左括号入栈放下标,出栈弹出栈顶的下标去匹配掉
	// 弹出后栈空,最后一个未被匹配的右括号下标即为当前下标
	// 弹出后栈不为空，比较大小

	// 为了初始化处理空的情况方便,给栈底初始化一个值为-1
	maxAns := 0
	stack := []int{}
	stack = append(stack, -1)
	for index := range s {
		if s[index] == '(' {
			stack = append(stack, index)
		} else {
			stack = stack[:len(stack)-1]
			// 栈空更新栈底下标
			if len(stack) == 0 {
				stack = append(stack, index)
			} else {
				// 栈不为空,说明有破坏匹配的,比较下大小
				maxAns = max(maxAns, index-stack[len(stack)-1])
			}
		}
	}
	return maxAns
}

func generateParenthesis(n int) []string {
	arr := new([]string)
	parenthesis(n, n, "", arr)
	return *arr
}

func parenthesis(left, right int, tmp string, res *[]string) {
	if right == 0 {
		*res = append(*res, tmp)
		return
	}
	// 生成左括号
	if left > 0 {
		parenthesis(left-1, right, tmp+"(", res)
	}
	// 生成友括号
	if right > left {
		parenthesis(left, right-1, tmp+")", res)
	}

}

//下一个排列
func nextPermutation(nums []int) {
	if nums == nil || len(nums) == 1 {
		return
	}
	i, j := len(nums)-2, len(nums)-1
	//  选取距离当前一个数下一个大的数
	// 4,2,0,2,3,2,0
	for i >= 0 && nums[i] >= nums[j] {
		i--
		j--
	}
	//  这里获取出了两个数的位置
	k := len(nums) - 1
	if i >= 0 {
		// 获取出来的数进行对比
		for nums[i] >= nums[k] {
			k--
		}
		nums[i], nums[k] = nums[k], nums[i]
	}
	sort.Ints(nums[i+1:])
	fmt.Println(nums)

}

// Swap 交换两个整数
func Swap(a, b int) {
	a = a ^ b
	b = a ^ b
	a = a ^ b
}

//串联所有单词的子串
func findSubstring(s string, words []string) []int {
	// 存放出现的位置
	arr := []int{}
	length := len(words[0])
	sum := length * len(words)
	// 长度不满足
	if len(s) < sum {
		return nil
	}
	// 开始存放每个单词出现的次数
	m1 := make(map[string]int)
	for _, value := range words {
		// 查看第一个map中是否出现过该单词
		if v1, ok := m1[value]; ok {
			m1[value] = v1 + 1
		} else {
			m1[value] = 1
		}
	}
	for i := 0; i < len(s)-sum+1; i++ {
		// 判断新的单词出现的数量，
		m2 := make(map[string]int)
		num := 0
		for num < len(words) {
			//	 获取当前word的长度
			word := s[i+length*num : i+(num+1)*length]
			// 判断字符是否存在在map中
			if value, ok := m1[word]; ok {
				//	 标明数组存在里面
				if val, flag := m2[word]; flag {
					if val >= value {
						break
					}
					m2[word] = val + 1
				} else {
					m2[word] = 1
				}
			} else {
				break
			}
			num++
		}
		if num == len(words) {
			arr = append(arr, i)
		}
	}
	return arr
}

// 两数相除
func divide(dividend int, divisor int) int {
	if dividend == 0 {
		return 0
	}
	if dividend == math.MinInt32 && divisor == -1 {
		return math.MaxInt32
	}

	diffSign := false
	if (dividend < 0) != (divisor < 0) {
		diffSign = true
	}

	i, j, sum := 0, 0, 0
	for {
		tmp := 0
		if diffSign {
			tmp = sum - divisor<<j
		} else {
			tmp = sum + divisor<<j
		}
		if (dividend > 0 && tmp > dividend) || (dividend < 0 && tmp < dividend) {
			if j == 0 {
				break
			}
			j-- //步长减半
			continue
		}
		sum = tmp
		i += 1 << j
		j++ //步长加倍
	}

	if diffSign {
		return -i
	}
	return i

}

// 查找字符串
func strStr(haystack string, needle string) int {
	if len(needle) == 0 { //若模式串为空串
		return 0
	}
	if len(haystack) == 0 || len(haystack) < len(needle) {
		return -1
	}
	index := 0
	j := 0
	for i := 0; i <= len(haystack)-len(needle); i++ {
		// 当第一个字符匹配的时候
		for j = 0; j < len(needle); j++ {
			if haystack[i+j] == needle[j] {
				index = i
			} else {
				break
			}
		}
		if j == len(needle) {
			return index
		}
	}
	return -1
}

// 删除数组
func removeElement(nums []int, val int) int {
	lenght := len(nums)
	if lenght == 0 {
		return 0
	}
	res := 0
	for i := 0; i < lenght; i++ {
		// 不等将数据修改到数组，相等直接跳过
		if nums[i] != val {
			nums[res] = nums[i]
			res++
		}
	}
	return res
}

// 删除有序数组的重复项
func removeDuplicates(nums []int) int {
	n := len(nums)
	if n < 2 {
		return n
	}
	left, right := 1, 1
	for right < n {
		if nums[right] != nums[right-1] {
			nums[left] = nums[right]
			left++
		}
		right++
	}
	return left
}

// 分为 反转部分 待反转部分 未反转部分
func reverseKGroup(head *ListNode, k int) *ListNode {
	if k == 1 || head == nil || head.Next == nil {
		return head
	}
	// 前置节点
	prev := &ListNode{0, nil}
	// 用于返回反转完成的链表
	prev.Next = head
	//	前置节点
	pre := prev
	// 待反转的节点
	left, right := head, head
	//	 初始化指针
	// 前面待反转的节点
	for i := 0; i < k-1; i++ {
		right = right.Next
	}
	//	 记录步长
	count := 0
	for right != nil {
		if count%k == 0 {
			left, right = reverse(left, right)
			// 新的前置节点
			pre.Next = left
		}
		pre = pre.Next
		left = left.Next
		right = right.Next
		count++
	}
	return prev.Next
}

// 链表反转 反转新的链表
func reverse(head, tail *ListNode) (*ListNode, *ListNode) {
	prev := tail.Next
	cur := head
	for prev != tail {
		// 获取left 节点之后的数据
		temp := cur
		// 将left的第一个节点和第二个节点分开
		cur = cur.Next
		// 将right后面的节点放在left第一个节点后面
		temp.Next = prev
		// 组装新的right节点的数据
		prev = temp
	}
	return prev, head
}

// 单链表反转
func reverseList(head *ListNode) *ListNode {
	prev := &ListNode{}
	curr := head
	for curr != nil {
		// 将当前节点的下一个节点存储起来，避免丢失
		nextTemp := curr.Next
		// 这里表示当前节点的下一个节点成为当前节点的上一个节点
		curr.Next = prev
		// 将curr节点变为头结点
		prev = curr
		// 进行下一个节点进行循环
		curr = nextTemp
	}
	return prev
}

func swapPairs(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	h := &ListNode{0, head}
	p := h
	// 真正的头结点
	cur := h.Next
	//	 第一次表示第二个节点和第三个节点为空
	for cur != nil && cur.Next != nil {
		// 将第二个节点放在虚拟头结点后面
		p.Next = cur.Next
		// 将第三个节点放入第一个节点后面
		cur.Next = cur.Next.Next
		// 将第一个节点放在转换后的第二个节点
		p.Next.Next = cur
		// 然后将第三个节点设置虚拟头结点继续后面的操作
		p = cur
		cur = cur.Next
	}
	return h.Next
}

func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	}
	tem := lists[0]
	for i := 1; i < len(lists); i++ {
		tem = mergeTwoLists(tem, lists[i])
	}
	return tem
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	tme := &ListNode{}
	next := tme
	for l1 != nil && l2 != nil {
		if l1.Val <= l2.Val {
			tme.Next = l1
			l1 = l1.Next
		} else {
			tme.Next = l2
			l2 = l2.Next
		}
		tme = tme.Next
	}
	if l1 == nil {
		tme.Next = l2
	}
	if l2 == nil {
		tme.Next = l1
	}
	return next.Next
}

func isValid(s string) bool {
	if len(s) == 0 {
		return false
	}
	// 创建一个栈 ，
	var stack []byte

	tmp := make(map[byte]byte)
	tmp[')'] = '('
	tmp['}'] = '{'
	tmp[']'] = '['
	for i := 0; i < len(s); i++ {
		if s[i] == '(' || s[i] == '{' || s[i] == '[' {
			stack = append(stack, s[i])
		} else {
			if len(stack) == 0 {
				return false
			}
			if tmp[s[i]] != stack[len(stack)-1] {
				return false
			}
			stack = stack[:len(stack)-1]
		}

	}
	return len(stack) == 0
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	tmp := &ListNode{0, head}
	second := tmp
	first := head
	for i := 0; i < n; i++ {
		if first == nil {
			return nil
		}
		first = first.Next
	}
	for first != nil {
		second = second.Next
		first = first.Next
	}
	second.Next = second.Next.Next
	return tmp.Next

}

type ListNode struct {
	Val  int
	Next *ListNode
}

// 四数之和
func fourSum(nums []int, target int) [][]int {
	sums := [][]int{}
	sort.Ints(nums)

	for i := 0; i < len(nums)-3; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		for j := i + 1; j < len(nums)-2; j++ {
			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}
			l, r := j+1, len(nums)-1
			for l < r {
				n1, n2, n3, n4 := nums[i], nums[j], nums[l], nums[r]
				if n1+n2+n3+n4 < target {
					l++
				} else if n1+n2+n3+n4 > target {
					r--
				} else if n1+n2+n3+n4 == target {
					sums = append(sums, []int{n1, n2, n3, n4})
					for l < r && n3 == nums[l+1] {
						l++
					}
					for l < r && n4 == nums[r-1] {
						r--
					}
					l++
					r--
				}
			}
		}
	}
	return sums
}

func letterCombinations(digits string) []string {

	m := make(map[byte][]string)
	m['2'] = []string{"a", "b", "c"}
	m['3'] = []string{"d", "e", "f"}
	m['4'] = []string{"g", "h", "i"}
	m['5'] = []string{"j", "k", "l"}
	m['6'] = []string{"m", "n", "o"}
	m['7'] = []string{"p", "q", "r", "s"}
	m['8'] = []string{"t", "u", "v"}
	m['9'] = []string{"w", "x", "y", "z"}

	var ret []string
	for i := 0; i < len(digits); i++ {
		ret = getCombination(ret, m[digits[i]])
	}
	return ret

}

func getCombination(a, b []string) []string {

	if len(a) == 0 {
		return b
	}
	if len(b) == 0 {
		return a
	}
	var ret []string
	// 将两个都遍历一下，然后将数据叠加
	for _, value := range a {
		for _, v := range b {
			ret = append(ret, value+v)
		}
	}
	return ret
}

//  三数之和
func threeSumClosest(nums []int, target int) int {
	res := nums[0] + nums[1] + nums[2]
	sort.Ints(nums)
	for i := 0; i < len(nums)-2; i++ {
		l, r := i+1, len(nums)-1
		for l < r {
			tmp := nums[i] + nums[l] + nums[r]
			if tmp > target {
				r--
			} else if tmp < target {
				l++
			} else {
				return target
			}
			if distance(tmp, target) < distance(res, target) {
				res = tmp
			}
		}
	}
	return res
}

func distance(a, b int) int {
	if a < b {
		return b - a
	}
	return a - b
}

// 罗马文转数字
func romanToInt(s string) int {
	if len(s) == 0 {
		return 0
	}
	m := make(map[string]int)
	m["I"] = 1
	m["V"] = 5
	m["X"] = 10
	m["L"] = 50
	m["C"] = 100
	m["D"] = 500
	m["M"] = 1000
	num := 0

	// 从做到又，如果右边的比左边的大，那么代表这个是一个组合的数
	for i := 0; i < len(s)-1; i++ {
		if m[string(s[i])] < m[string(s[i+1])] {
			// 左边比右边的小，所以减去这个数
			num -= m[string(s[i])]
		} else {
			num += m[string(s[i])]
		}
	}
	// 将最后的一个数加上
	num += m[string(s[len(s)-1])]
	return num
}

// 整数转罗马数字
func intToRoman(num int) string {
	if num == 0 {
		return ""
	}
	m := make(map[int]string)
	m[1] = "I"
	m[4] = "IV"
	m[5] = "V"
	m[9] = "IX"
	m[10] = "X"
	m[40] = "XL"
	m[50] = "L"
	m[90] = "XC"
	m[100] = "C"
	m[400] = "CD"
	m[500] = "D"
	m[900] = "CM"
	m[1000] = "M"
	res := ""
	index := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
	for _, index := range index {
		if index <= num {
			times := num / index
			num %= index
			for i := 0; i < times; i++ {
				res += m[index]
			}
		}
	}
	return res
}

// 连续子数组的乘积
func maxProduct(nums []int) int {

	if len(nums) == 1 {
		return nums[0]
	}
	maxn, minn := 1, 1
	res := nums[0]
	for i := 0; i < len(nums); i++ {
		permax := maxn
		// 如果是整数的情况
		maxn = max(max(permax*nums[i], minn*nums[i]), nums[i])
		// 如果是负数的情况
		minn = min(min(permax*nums[i], minn*nums[i]), nums[i])
		res = max(maxn, res)
	}
	return res
}

func findCheapestPrice(n int, flights [][]int, src int, dst int, k int) int {
	// dp【i】表示经过k个站到达城市i
	dp := make([][]int, n)
	// 初始化db
	for i := 0; i < n; i++ {
		//	这里是从 src到dst，经过最多经过k城市的次数
		dp[i] = make([]int, k+1)
		for j := 0; j <= k; j++ {
			//	-1代表的不可到达，初始化时认为所有的到达的城市i经过k个站都是，不可到达，办不到的
			dp[i][j] = -1
		}
	}
	for _, v := range flights {
		fmt.Println(v)
		if v[0] == src {
			dp[v[1]][0] = v[2]
		}
	}
	for i := 0; i < k; i++ {
		dp[src][i] = 0
	}

	fmt.Println(dp)
	return 0
}

func minCount(coins []int) int {
	num := 0
	for _, value := range coins {
		sum := value % 2
		if sum == 1 {
			num += 1 + value/2
		} else {
			num += value / 2
		}
	}
	return num
}

func heightChecker(heights []int) int {
	num := 0
	nums := []int{}
	nums = append(nums, heights...)
	sort.Ints(heights)
	for i := 0; i < len(heights); i++ {
		if nums[i] != heights[i] {
			num++
		}
	}
	return num
}

func foo(value int) int {
	var val int = 11
	return val

}

func getMaximumGenerated(n int) int {
	if n < 2 {
		return n
	}
	nums := make([]int, 300)
	nums[0] = 0
	nums[1] = 1
	res := 0
	for i := 2; i <= n; i++ {

		nums[i*2] = nums[i]
		nums[i*2+1] = nums[i] + nums[i+1]
		if nums[i] > res {
			res = nums[i]
		}

	}
	return res
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
