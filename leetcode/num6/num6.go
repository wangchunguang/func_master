package num6

import (
	"fmt"
	"func_master"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

type ListNode struct {
	Val  int
	Next *ListNode
}
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func AddListNode() *ListNode {
	//head := &ListNode{1, &ListNode{2, &ListNode{3,&ListNode{4,&ListNode{5,nil}}}}}
	head := &ListNode{1, &ListNode{2, &ListNode{6,
		&ListNode{3, &ListNode{4, &ListNode{5, &ListNode{6, nil}}}}}}}
	return head
}
func AddListNode1() *ListNode {
	//head := &ListNode{1, &ListNode{2, &ListNode{3,&ListNode{4,&ListNode{5,nil}}}}}
	head := &ListNode{3, &ListNode{2, &ListNode{4, nil}}}
	return head
}

func AddTreeNode() *TreeNode {
	return &TreeNode{6,
		&TreeNode{2, &TreeNode{0, nil, nil}, &TreeNode{4, &TreeNode{3, nil, nil}, &TreeNode{5, nil, nil}}},
		&TreeNode{8, &TreeNode{7, nil, nil}, &TreeNode{9, nil, nil}}}

}

func AddNode() *Node {

	node := &Node{1, &Node{2, &Node{6, &Node{3, &Node{4, nil, nil}, nil}, nil}, nil}, nil}

	return node
}

// 300. 最长递增子序列
func lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	num := 1
	for i := 1; i < len(nums); i++ {
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[j]+1, dp[i])
				num = max(num, dp[i])
			}
		}
	}
	return num
}

//299. 猜数字游戏
func getHint(secret string, guess string) string {
	m := make(map[byte]int)
	se := len(secret)
	for i := 0; i < se; i++ {
		m[secret[i]]++
	}
	A, B := 0, 0
	m1 := make(map[int]int)
	// 先将相等的计算出来
	for i := 0; i < se; i++ {
		if secret[i] == guess[i] {
			m[secret[i]]--
			m1[i]++
			A++
			if m[secret[i]] == 0 {
				delete(m, secret[i])
			}
		}
	}
	for i := 0; i < se; i++ {
		if secret[i] != guess[i] && m[guess[i]] != 0 && m1[i] == 0 {
			B++
			m[guess[i]]--
			if m[guess[i]] == 0 {
				delete(m, guess[i])
			}
		}
	}
	return strconv.Itoa(A) + "A" + strconv.Itoa(B) + "B"

}

// 297. 二叉树的序列化与反序列化
type Codec struct {
	Tree *TreeNode
}

func Constructor_297() Codec {
	return Codec{&TreeNode{}}
}

//BFS实现
// Serializes a tree to a single string.
func (this *Codec) serializeBFS(root *TreeNode) string {
	arr := []*TreeNode{root}
	str := []string{}
	for len(arr) != 0 {
		node := arr[0]
		arr = arr[1:]
		if node != nil {
			str = append(str, strconv.Itoa(node.Val))
			arr = append(arr, node.Left)
			arr = append(arr, node.Right)
		} else {
			str = append(str, "+")
		}
	}
	return strings.Join(str, ",")
}

// Deserializes your encoded data to tree.
func (this *Codec) deserializeBFS(data string) *TreeNode {
	split := strings.Split(data, ",")
	// 先将根节点提取出来
	if split[0] == "+" {
		return nil
	}
	atoi, _ := strconv.Atoi(split[0])
	split = split[1:]
	root := &TreeNode{Val: atoi}
	queue := []*TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		str := split[0]
		queue = queue[1:]
		split = split[1:]
		// 先创建左子树
		if str != "+" {
			num, _ := strconv.Atoi(str)
			left := &TreeNode{Val: num}
			node.Left = left
			queue = append(queue, left)
		}
		str = split[0]
		split = split[1:]
		if str != "+" {
			num, _ := strconv.Atoi(str)
			Right := &TreeNode{Val: num}
			node.Right = Right
			queue = append(queue, Right)
		}
	}
	return root
}

// DFS实现

// Serializes a tree to a single string.
func (this *Codec) serializeDFS(root *TreeNode) string {
	if root == nil {
		return "X"
	}
	return strconv.Itoa(root.Val) + "," + this.serializeDFS(root.Left) + "," + this.serializeDFS(root.Right)
}

// Deserializes your encoded data to tree.
func (this *Codec) deserializeDFS(data string) *TreeNode {
	split := strings.Split(data, ",")
	return buildTree(&split)
}

func buildTree(list *[]string) *TreeNode {
	val := (*list)[0]
	*list = (*list)[1:]
	if val == "X" {
		return nil
	}

	atoi, _ := strconv.Atoi(val)
	root := &TreeNode{Val: atoi}
	root.Left = buildTree(list)
	root.Right = buildTree(list)
	return root
}

type MedianFinder struct {
	// 大根堆 存储所有元素较小的一部分
	MaxHeap *func_master.Heap
	// 小根堆 存储所有元素较大的一部分
	MinHeap *func_master.Heap
}

//func Constructor() MedianFinder {
//	return MedianFinder{func_master.NewHeap(), func_master.NewHeap()}
//}

func (this *MedianFinder) AddNum(num int) {
	// 小于就直接进小根堆 或者等于0的时候，如果传进来的数 大于小根堆的堆顶，那么就直接存入
	// num大于小根堆的堆顶，表示可以存入小根堆的数据
	// 大根堆是 父节点大于或者等于子节点 小根堆表示父节点小于或者等于子节点
	if this.MinHeap.Count() == 0 || num > this.MinHeap.Pop() {
		this.MinHeap.PushMin(num)
		if (this.MinHeap.Count() - this.MaxHeap.Count()) > 1 {
			this.MaxHeap.PushMax(this.MinHeap.DelMin())
		}
	} else {
		this.MaxHeap.PushMax(num)
		if (this.MaxHeap.Count() - this.MinHeap.Count()) > 0 {
			this.MinHeap.PushMin(this.MaxHeap.DelMax())
		}
	}
}

func (this *MedianFinder) FindMedian() float64 {
	if this.MinHeap.Count() > this.MaxHeap.Count() {
		return float64(this.MinHeap.Pop())
	}
	avg := (float64(this.MaxHeap.Pop() + this.MinHeap.Pop())) / 2
	return avg
}

// 292. Nim 游戏
func canWinNim(n int) bool {
	if n%4 == 0 {
		return true
	}
	return false
}

// 290. 单词规律
func wordPattern(pattern string, s string) bool {
	words := strings.Split(s, " ")
	// 长度不一样
	if len(words) != len(pattern) {
		return false
	}
	// 将数据一一对应
	p_map := make(map[byte]string)
	s_map := make(map[string]byte)
	for key, val := range words {
		// 判断数据是否存在
		if p_map[pattern[key]] != "" && p_map[pattern[key]] != val || s_map[val] > 0 && s_map[val] != pattern[key] {
			return false
		}
		p_map[pattern[key]] = val
		s_map[val] = pattern[key]
	}
	return true
}

// 289. 生命游戏
func gameOfLife(board [][]int) {
	//	 定义这个数组，存储的坐标点，表示borad的位置数据相反
	arr := [][]int{}
	m, n := len(board), len(board[0])
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			count := 0
			if j > 0 {
				if board[i][j-1] == 1 {
					count++
				}
				if i > 0 && board[i-1][j-1] == 1 {
					count++
				}
				if i < m-1 && board[i+1][j-1] == 1 {
					count++
				}
			}
			if j < n-1 {
				if board[i][j+1] == 1 {
					count++
				}
				if i > 0 && board[i-1][j+1] == 1 {
					count++
				}
				if i < m-1 && board[i+1][j+1] == 1 {
					count++
				}
			}
			if i > 0 && board[i-1][j] == 1 {
				count++
			}
			if i < m-1 && board[i+1][j] == 1 {
				count++
			}
			if (count < 2 || count > 3) && board[i][j] == 1 {
				arr = append(arr, []int{i, j})
			} else if count == 3 && board[i][j] == 0 {
				arr = append(arr, []int{i, j})
			}
		}
	}
	for _, value := range arr {
		x, y := value[0], value[1]
		if board[x][y] == 1 {
			board[x][y] = 0
		} else {
			board[x][y] = 1
		}
	}
}

// 287 寻找重复数
func findDuplicate(nums []int) int {
	// 采用环形链表的做法
	l, r := 0, 0
	// 获取一次跳两个数的最后节点
	for {
		l = nums[l]
		r = nums[r]
		r = nums[r]
		if r == l {
			break
		}
	}
	r = 0
	for {
		r = nums[r]
		l = nums[l]
		if r == l {
			break
		}
	}
	return r
}

//283. 移动零
func moveZeroes(nums []int) {
	j := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[j] = nums[i]
			j++
		}
	}
	for i := j; i < len(nums); i++ {
		nums[i] = 0
	}
}

// 282. 给表达式添加运算符
func addOperators(num string, target int) []string {
	str := []string{}
	lenght := len(num)
	addOperators_dfs(num, "", &str, 0, 0, target, lenght, 0)
	return str
}

/**
num 表示原始数组
path表示当前组装的数据
str 表示返回的数据
index 表示已经编写了多少位
target表示需要对比的数
value 表示当前计算的值
prve 上次的值
*/
func addOperators_dfs(num, path string, str *[]string, index, value, target, lenght, prve int) {
	if index == lenght {
		if target == value {
			*str = append(*str, path)
		}
		return
	}
	for i := index; i < lenght; i++ {
		// 只有当i向后面走 并且当前位置为0的时候才会跳过，因为没有0开头的数
		// 获取首位不为0，并且一直延伸到后面的数
		if index != i && num[index] == '0' {
			break
		}
		next, _ := strconv.Atoi(num[index : i+1])
		nestStr := num[index : i+1]
		// 当等于0位的时候
		if index == 0 {
			addOperators_dfs(num, ""+nestStr, str, i+1, next, target, lenght, next)
		} else {
			addOperators_dfs(num, path+"+"+nestStr, str, i+1, value+next, target, lenght, next)
			addOperators_dfs(num, path+"-"+nestStr, str, i+1, value-next, target, lenght, -next)
			x := next * prve
			addOperators_dfs(num, path+"*"+nestStr, str, i+1, value-prve+x, target, lenght, x)
		}
	}
}

// Sign signNum表示当月签到多少天，num表示当前多少号
func Sign(signNum *uint32, num int) {
	// 检查距离当前有多少天未签到
	formatInt := strconv.FormatInt(int64(*signNum), 2)
	l1 := num - len(formatInt) + 1
	*signNum = *signNum << l1
	*signNum += 1
	fmt.Println(strconv.FormatInt(int64(*signNum), 2))
}

// 279. 完全平方数
func numSquares(n int) int {
	dp := make([]int, n+1)
	for i := 0; i <= n; i++ {
		dp[i] = i
		for j := 0; j*j <= i; j++ {
			dp[i] = min(dp[i], dp[i-j*j]+1)
		}
	}
	return dp[n]
}

// 278. 第一个错误的版本
//func isBadVersion(version int) bool
//
//func firstBadVersion(n int) int {
//	m := 0
//	for m <= n {
//		tmp := m + (n-m)/2
//		if isBadVersion(tmp) {
//			n = tmp - 1
//		} else {
//			m = tmp + 1
//		}
//	}
//	return m
//}

//274. H 指数
func hIndex(citations []int) int {

	sort.Slice(citations, func(i, j int) bool {
		return citations[i] < citations[j]
	})
	// 从左到右线性遍历
	n := len(citations)
	for i := 0; i < n; i++ {
		if citations[i] >= n-i {
			return n - i
		}
	}
	return 0
}

var num2str_small = []string{
	"Zero",
	"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
	"Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen",
}
var num2str_medium = []string{
	"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety",
}

var num2str_large = []string{
	"Billion", "Million", "Thousand", "",
}

// 273. 整数转换英文表示
func numberToWords(num int) string {
	if num == 0 {
		return num2str_small[0]
	}
	sb := strings.Builder{}
	for i, j := 1000000000, 0; i > 0; i, j = i/1000, j+1 {
		if num < i {
			continue
		}
		sb.WriteString(numToStr(num/i) + num2str_large[j] + " ")
		num %= i
	}
	return strings.TrimSpace(sb.String())

}

func numToStr(x int) (ans string) {
	if x >= 100 {
		ans += num2str_small[x/100] + " Hundred "
		x %= 100
	}
	if x >= 20 {
		ans += num2str_medium[x/10] + " "
		x %= 10
	}
	if x != 0 {
		ans += num2str_small[x] + " "
	}
	return
}

// 268. 丢失的数字
func missingNumber(nums []int) int {

	sort.Ints(nums)
	if nums[0] == 1 {
		return 0
	}
	num := 1
	for num < len(nums) {
		if nums[num] != nums[num-1]+1 {
			return nums[num]
		}
		num++
	}
	return num
}

// 264. 丑数 II
func nthUglyNumber(n int) int {

	dp := make([]int, n)
	dp[0] = 1
	p2, p3, p5 := 0, 0, 0
	for i := 1; i < n; i++ {
		v2 := dp[p2] * 2
		v3 := dp[p3] * 3
		v5 := dp[p5] * 5
		v := min(v2, min(v3, v5))

		dp[i] = v
		if v == v2 {
			p2++
		}
		if v == v3 {
			p3++
		}
		if v == v5 {
			p5++
		}
	}
	return dp[n-1]
}

// 263. 丑数
func isUgly(n int) bool {
	if n == 0 {
		return false
	}
	for n != 1 {
		if n%2 == 0 {
			n /= 2
		} else if n%3 == 0 {
			n /= 3
		} else if n%5 == 0 {
			n /= 5
		} else {
			return false
		}
	}
	return true

}

//260. 只出现一次的数字 III
func singleNumber(nums []int) []int {
	m := make(map[int]int)
	for _, value := range nums {
		if _, ok := m[value]; ok {
			delete(m, value)
		} else {
			m[value]++
		}
	}
	arr := []int{}
	for key, _ := range m {
		arr = append(arr, key)
	}
	return arr

}

// 258. 各位相加
func addDigits(num int) int {
	for num >= 10 {
		num = num%10 + num/10
	}
	return num
}

//257. 二叉树的所有路径
func binaryTreePaths(root *TreeNode) []string {
	arr := []string{}
	binaryTreePaths_dfs(root, &arr, "")
	return arr
}
func binaryTreePaths_dfs(root *TreeNode, arr *[]string, str string) {
	if root == nil {
		return
	}
	if root.Left == nil && root.Right == nil {
		str += strconv.Itoa(root.Val)
		*arr = append(*arr, str)
		return
	}

	str += strconv.Itoa(root.Val) + "->"
	binaryTreePaths_dfs(root.Left, arr, str)
	binaryTreePaths_dfs(root.Right, arr, str)
}

// 242. 有效的字母异位词
func isAnagram(s string, t string) bool {
	l1, l2 := len(s), len(t)
	m := make(map[int32]int)
	if l1 != l2 {
		return false
	}
	for _, val := range t {
		m[val]++
	}
	for _, val := range s {
		if _, ok := m[val]; !ok {
			return false
		}
		m[val]--
		if m[val] < 0 {
			return false
		}
	}
	return true
}

//241. 为运算表达式设计优先级
func diffWaysToCompute(expression string) []int {
	//	 判断是否全是数字
	if isDigit(expression) {
		atoi, _ := strconv.Atoi(expression)
		return []int{atoi}
	}
	var res []int
	for index, c := range expression {
		tmp := string(c)
		if tmp == "+" || tmp == "-" || tmp == "*" {
			//	 采用分治法计算两边的数据
			left := diffWaysToCompute(expression[:index])
			right := diffWaysToCompute(expression[index+1:])
			for _, leftNum := range left {
				for _, rightNum := range right {
					var addNum int
					if tmp == "+" {
						addNum = leftNum + rightNum
					} else if tmp == "-" {
						addNum = leftNum - rightNum
					} else {
						addNum = leftNum * rightNum
					}
					res = append(res, addNum)
				}
			}

		}
	}
	return res
}

func isDigit(input string) bool {
	_, err := strconv.Atoi(input)
	if err != nil {
		return false
	}
	return true
}

// 240. 搜索二维矩阵 II
func searchMatrix(matrix [][]int, target int) bool {
	for i := 0; i < len(matrix); i++ {
		if binarySearch(target, matrix[i]) != -1 {
			return true
		}
	}
	return false
}

// 二分查找非递归实现
func binarySearch(target int, nums []int) int {
	left := 0
	right := len(nums) - 1
	for left <= right {
		mid := left + (right-left)/2
		if target == nums[mid] {
			return mid
		}
		if target > nums[mid] {
			left = mid + 1
			continue
		}
		if target < nums[mid] {
			right = mid - 1
			continue
		}
	}
	return -1
}

// 239. 滑动窗口最大值
func maxSlidingWindow(nums []int, k int) []int {
	queue := func_master.MonotonousQueue{}
	length := len(nums)
	res := make([]int, 0)
	// 先将前k个元素放入队列
	for i := 0; i < k; i++ {
		queue.Push(nums[i])
	}
	// 记录前k个元素的最大值
	res = append(res, queue.Front())
	for i := k; i < length; i++ {
		// 滑动窗口移除最前面的元素
		queue.Pop(nums[i-k])
		// 滑动窗口添加最后面的元素
		queue.Push(nums[i])
		// 记录最大值
		res = append(res, queue.Front())
	}
	return res
}

// 238. 除自身以外数组的乘积
func productExceptSelf(nums []int) []int {
	lenght := len(nums)
	//	计算每个数组左边的乘积
	dp_left := make([]int, lenght)
	dp_left[0] = 1
	dp_left[1] = nums[0]
	for i := 2; i < lenght; i++ {
		dp_left[i] = nums[i-1] * dp_left[i-1]
	}
	// 计算右边的乘积
	dp_right := make([]int, lenght)
	dp_right[lenght-1] = 1
	for i := lenght - 2; i >= 0; i-- {
		dp_right[i] = nums[i+1] * dp_right[i+1]
	}
	arr := make([]int, lenght)
	for i := 0; i < lenght; i++ {
		arr[i] = dp_left[i] * dp_right[i]
	}
	return arr
}

// 237. 删除链表中的节点
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// 236. 二叉树的最近公共祖先
func lowestCommonAncestor_236(root, p, q *TreeNode) *TreeNode {
	if root == nil || p == root || q == root {
		return root
	}

	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	// 当左子树右子树都为空的时候 表示都不包含pq 返回nil
	if right == nil && left == nil {
		return nil
	}
	if left == nil {
		return right
	}
	if right == nil {
		return left
	}
	return root
}

// 235. 二叉搜索树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val > p.Val && root.Val > q.Val {
		return lowestCommonAncestor(root.Left, p, q)
	}
	if root.Val < p.Val && root.Val < q.Val {
		return lowestCommonAncestor(root.Right, p, q)
	}
	return root
}

// 234. 回文链表
func isPalindrome(head *ListNode) bool {
	arr := []int{}
	for head != nil {
		arr = append(arr, head.Val)
		head = head.Next
	}

	l, r := 0, len(arr)
	for l < r {
		if arr[l] != arr[r] {
			return false
		}
		l++
		r--
	}
	return true
}

// 231. 2 的幂
func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

// 230. 二叉搜索树中第K小的元素
func kthSmallest(root *TreeNode, k int) int {
	if root == nil {
		return 0
	}
	arr := []int{}
	kthSmallest_dfs(root, &arr)
	return arr[k-1]
}

func kthSmallest_dfs(root *TreeNode, arr *[]int) []int {
	if root == nil {
		return nil
	}
	kthSmallest_dfs(root.Left, arr)
	*arr = append(*arr, root.Val)
	kthSmallest_dfs(root.Right, arr)
	return *arr
}

// 229. 求众数 II
func majorityElement(nums []int) []int {
	num := len(nums) / 3
	num_map := make(map[int]int)
	arr := []int{}
	for i := 0; i < len(nums); i++ {
		num_map[nums[i]]++
		if num_map[nums[i]] > num {
			arr = append(arr, nums[i])
		}
	}
	for key, value := range num_map {
		if value > num {
			arr = append(arr, key)
		}
	}
	return arr
}

// 228. 汇总区间
func summaryRanges(nums []int) []string {
	//	 总长度
	n := len(nums)
	// 双指针计算
	l, r := 0, 0
	res := []string{}
	for r < n {
		for r < n-1 && nums[r+1] == nums[r]+1 {
			r++
		}
		// 两个数据一样
		if l == r {
			res = append(res, strconv.Itoa(nums[r]))
		} else {
			res = append(res, strconv.Itoa(nums[l])+"->"+strconv.Itoa(nums[r]))
		}
		r++
		l = r
	}

	return res
}

// 227. 基本计算器 II
func calculate_227(s string) int {
	stack := []int{}
	// 表示上一个出现的符号
	sign := '+'
	res := 0
	num := 0
	s = s + "+"
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			continue
		}
		// 如果为数字的时候
		if s[i] >= 48 && s[i] <= 57 {
			num = num*10 + int(s[i]-'0')
		} else {
			switch sign {
			case '+':
				stack = append(stack, num)
			case '-':
				stack = append(stack, -num)
			case '/':
				stack[len(stack)-1] = stack[len(stack)-1] / num
			case '*':
				stack[len(stack)-1] = stack[len(stack)-1] * num
			}
			sign = int32(s[i])
			num = 0
		}
	}
	for i := 0; i < len(stack); i++ {
		res += stack[i]
	}

	return res
}

// 226. 翻转二叉树
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	left := invertTree(root.Left)
	right := invertTree(root.Right)
	root.Left, root.Right = right, left
	return root
}

// 224. 基本计算器
func calculate(s string) int {
	// 采用分层计算，一个完整的（）里面计算出来一个数
	//	 （左边计算出来一个数
	stack := make([]int, 0)
	//	 当前的符号 1为+ -1 为-
	symbol := 1
	//	 初始化的数据
	//计算每一层的数据
	num := 0
	// 需要返回的数
	res := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '1', '2', '3', '4', '5', '6', '7', '8', '9', '0':
			j := i
			num = 0
			for j < len(s) && unicode.IsDigit(rune(s[j])) {
				num = num*10 + int(s[j]-'0')
				j++
			}
			res += num * symbol
			i = j - 1
		case '(': // 当遇见（时，将数据先存储起来
			stack = append(stack, res)
			stack = append(stack, symbol)
			symbol = 1
			res = 0
		case ')':
			symbol = stack[len(stack)-1]
			pre := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			res = pre + symbol*res
		case '+':
			symbol = 1
		case '-':
			symbol = -1
		}

	}
	return res
}

// 223. 矩形面积
func computeArea(ax1 int, ay1 int, ax2 int, ay2 int, bx1 int, by1 int, bx2 int, by2 int) int {
	//两个面积 减去重叠的面积
	area := (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1)
	// 计算覆盖的面积
	// 计算x轴上有多长是覆盖的，右边取最小 左边取最大
	x := max(0, min(ax2, bx2)-max(ax1, bx1))
	// 计算y轴上覆盖多长，上面的取最大，下面的取最大
	y := max(0, min(ay2, by2)-max(ay1, by1))
	return area - x*y
}

func computeArea_abs(a, b int) int {
	return int((math.Abs(float64(a)) + math.Abs(float64(b))))
}

//给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	// dfs 递归的方式
	return countNodes_dfs(root)
	//	 寻常方式 采用宽度优先
	//num := 0
	//arr := []*TreeNode{root}
	//for len(arr) != 0 {
	//	node := arr[0]
	//	arr = arr[1:]
	//	if node.Left != nil {
	//		arr = append(arr, node.Left)
	//	}
	//	if node.Right != nil {
	//		arr = append(arr, node.Right)
	//	}
	//	num++
	//
	//}
	//return num
}

func countNodes_dfs(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := countNodes_dfs(root.Left)
	right := countNodes(root.Right)
	return left + right + 1
}

// 221. 最大正方形
func maximalSquare(matrix [][]byte) int {

	row, list := len(matrix), len(matrix[0])
	dp := make([][]int, row)
	for i := 0; i < row; i++ {
		dp[i] = make([]int, list)
	}
	//	 填充第一列 第一行
	// 如果出现过1 后面则 如果没有出现过1 那么就是0
	// 如果是1 表示面积为1 0表示面积为0
	max_num := 0
	for i := 0; i < row; i++ {
		if matrix[i][0] == '1' {
			dp[i][0] = 1
			max_num = 1
		} else {
			dp[i][0] = 0
		}
	}
	for i := 0; i < list; i++ {
		if matrix[0][i] == '1' {
			dp[0][i] = 1
			max_num = 1
		} else {
			dp[0][i] = 0
		}
	}
	for i := 1; i < row; i++ {
		for j := 1; j < list; j++ {
			if matrix[i][j] == '0' {
				continue
			}
			// 如果当前值为1 并且他的左边 上边左上方都为一样的，那么当前位置的值++
			if dp[i-1][j-1] == dp[i-1][j] && dp[i-1][j] == dp[i][j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
			}
			max_num = max(max_num, dp[i][j])
		}
	}
	return max_num * max_num
}

// 220. 存在重复元素 III
func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	for i := range nums {
		for j := i + 1; j <= i+k && j < len(nums); j++ {
			if abs(nums[i], nums[j]) <= t {
				return true
			}
		}
	}
	return false
}

func abs(a, b int) int {
	return int(math.Abs(float64(a - b)))
}

// 218. 天际线问题
func getSkyline(buildings [][]int) [][]int {
	var (
		res   = [][]int{}
		arr   = [][]int{}
		m_num = map[int]int{}
		pre   = -1
	)
	for _, value := range buildings {
		cur := [][]int{
			// 因为每一栋楼都是左边添加这个多的高度，右边减少这么多的高度，所以将数据存储起来
			{value[0], value[2]},
			{value[1], -value[2]},
		}
		arr = append(arr, cur...)
	}
	// [[2 10] [3 15] [5 12] [7 -15] [9 -10] [12 -12] [15 10] [19 8] [20 -10] [24 -8]]
	// [[2 -10] [3 -15] [5 -12] [7 15] [9 10] [12 12] [15 -10] [19 -8] [20 10] [24 8]]
	// 将数组进行遍历，第一位按照从大到小，第一位如果一样，第二位找按照从打到小
	arrSort(arr)
	for _, val := range arr {
		removeOrAdd(m_num, val[1])
		//	 获取map里面的最大值
		num := myMax(m_num)
		if pre != num {
			cur := []int{val[0], num}
			res = append(res, cur)
			pre = num
		}
	}
	return res
}
func myMax(m_num map[int]int) (num int) {
	for key, _ := range m_num {
		num = max(key, num)
	}
	return
}

func removeOrAdd(m_num map[int]int, num int) {
	if num < 0 {
		m_num[-num]--
		if m_num[-num] == 0 {
			delete(m_num, -num)
		}
	} else {
		m_num[num]++
	}
}

func arrSort(arr [][]int) {
	sort.Slice(arr, func(i, j int) bool {
		if arr[i][0] == arr[j][0] {
			return arr[i][1] > arr[j][1]
		} else {
			return arr[i][0] <= arr[j][0]
		}
	})

}

func IsDiffDay(now, old int64, timezone int) int {
	now += int64(timezone * 3600)
	old += int64(timezone * 3600)
	return int((now / 86400) - (old / 86400))
}

// 124 二叉树的最大路径和
func maxPathSum(root *TreeNode) int {
	max_sum := math.MinInt64
	maxPathSum_dfs(root, &max_sum)
	return max_sum
}

// 采用深度优先进行计算
func maxPathSum_dfs(root *TreeNode, max_num *int) int {
	if root == nil {
		return 0
	}
	//	 获取左边的最大参数
	left_dfs := maxPathSum_dfs(root.Left, max_num)
	right_dfs := maxPathSum_dfs(root.Right, max_num)
	//	 当前的值应该是左边 +右边 +当前的值
	sum := left_dfs + right_dfs + root.Val
	*max_num = max(sum, *max_num)
	//	 返回的参数是选取左边或者右边最大的一个数 继续比较
	return max(root.Val+max(left_dfs, right_dfs), 0)
}

// 219. 存在重复元素 II
func containsNearbyDuplicate(nums []int, k int) bool {
	m := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		// 表示这个数之前存在
		if value, ok := m[nums[i]]; ok {
			if i-value <= k {
				return true
			}
			m[nums[i]] = i
		} else {
			//	 表示这个数之前没有存在过
			m[nums[i]] = i
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
