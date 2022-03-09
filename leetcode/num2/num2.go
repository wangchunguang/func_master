package num2

import (
	"fmt"
	"func_master/leetcode/num1"
	"math"
	"sort"
	"strconv"
	"strings"
)

var result [][]int
var arr [][]int

// Pow(x,n)
func myPow(x float64, n int) float64 {
	//	 如果大于0的时候
	if n >= 0 {
		return quickMul(x, n)
	}
	// 小于0的时候
	return 1.0 / quickMul(x, -n)
}

func isNumber(s string) bool {
	if len(s) == 0 {
		return false
	}
	// 是否是数字
	isNum := false
	// 是否是小数点
	isDot := false
	// 是否是小e或者大E
	ise_or_E := false
	for i := range s {
		if s[i] >= '0' && s[i] <= '9' {
			// 出现过数字
			isNum = true
		} else if s[i] == '.' {
			// 是否出现过小数点或者是否出现过e，因为出现过小数点或者出过e，再出现小数点直接返回例如：99e2.5
			// e表示显示不完整的数，所以出现过e之后再出现小数点则会报错
			if isDot || ise_or_E {
				return false
			}
			isDot = true
		} else if s[i] == 'e' || s[i] == 'E' {
			// 没有出现过数字，或者已经出现过e直接返回
			if !isNum || ise_or_E {
				return false
			}
			ise_or_E = true
			// 出现e之后 都可以重新计算
			isNum = false
		} else if s[i] == '-' || s[i] == '+' {
			// 当不是第一位的时候，+或者-的上一位只能是e
			if i != 0 && s[i-1] != 'e' && s[i-1] != 'E' {
				return false
			}
		} else {
			return false
		}
	}
	return isNum
}

// 最小路径和
func minPathSum1(grid [][]int) int {
	//	多少行
	row_len := len(grid)
	// 多少列
	list_list := len(grid[0])
	dp := make([][]int, row_len)
	for i := 0; i < row_len; i++ {
		dp[i] = make([]int, list_list)
	}
	fmt.Println(dp)
	// 初始化最左上角的数据
	dp[0][0] = grid[0][0]
	for i := 0; i < row_len; i++ {
		for j := 0; j < list_list; j++ {
			if i == 0 && j == 0 {
				continue
			} else if i == 0 {
				dp[i][j] = dp[i][j-1] + grid[i][j]
			} else if j == 0 {
				dp[i][j] = dp[i-1][j] + grid[i][j]
			} else {
				dp[i][j] = min(dp[i-1][j]+grid[i][j], dp[i][j-1]+grid[i][j])
			}
		}
	}
	fmt.Println(dp)
	return dp[row_len-1][list_list-1]
}

// 最小路径和
func minPathSum(grid [][]int) int {
	//	多少行
	row_len := len(grid)
	// 多少列
	list_list := len(grid[0])
	dp := make([][]int, row_len)
	for i := 0; i < row_len; i++ {
		dp[i] = make([]int, list_list)
	}
	fmt.Println(dp)
	// 初始化最左上角的数据
	dp[0][0] = grid[0][0]
	// 初始化行
	for i := 1; i < list_list; i++ {
		dp[0][i] = dp[0][i-1] + grid[0][i]
	}
	// 初始化列
	for i := 1; i < row_len; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	fmt.Println(dp)
	for i := 1; i < row_len; i++ {
		for j := 1; j < list_list; j++ {
			dp[i][j] = min(dp[i-1][j]+grid[i][j], dp[i][j-1]+grid[i][j])
		}
	}
	fmt.Println(dp)
	return dp[row_len-1][list_list-1]
}

//不同路径2
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	// 行的长度
	row_len := len(obstacleGrid)
	//	 列的长度
	list_len := len(obstacleGrid[0])
	dp := make([][]int, row_len)
	for i := 0; i < row_len; i++ {
		dp[i] = make([]int, list_len)
	}
	// 先初始化第一行和第一列的数据 如果碰见不同的位置，将位置初始化为0
	// 当同一行遇见障碍物的时候 ，后续全是0 直接跳过
	for i := 0; i < list_len && obstacleGrid[0][i] == 0; i++ {
		dp[0][i] = 1
	}
	for i := 0; i < row_len && obstacleGrid[i][0] == 0; i++ {
		dp[i][0] = 1
	}
	fmt.Println(dp)
	for i := 1; i < row_len; i++ {
		for j := 1; j < list_len; j++ {
			if obstacleGrid[i][j] == 0 {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	fmt.Println(dp)
	return dp[row_len-1][list_len-1]
}

// 不同路径
func uniquePaths(m int, n int) int {
	arr := make([][]int, m)
	for i := 0; i < m; i++ {
		arr[i] = make([]int, n)
	}
	//	 先将每一行初始化
	for i := 0; i < m; i++ {
		arr[i][0] = 1
	}

	// 先将每一列进行初始化
	for j := 0; j < n; j++ {
		arr[0][j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			arr[i][j] = arr[i-1][j] + arr[i][j-1]
		}
	}
	return arr[m-1][n-1]
}

// 旋转链表
func rotateRight(head *num1.ListNode, k int) *num1.ListNode {
	//	 将链表移动
	if head == nil || k == 0 || head.Next == nil {
		return head
	}
	index := 1
	cur := head
	for cur.Next != nil {
		index++
		cur = cur.Next
	}
	if k >= index {
		k %= index
	}
	// 先将链表的长度取出来
	for i := 0; i < k; i++ {
		// 先将头结点提取出来
		prev := head
		left := &num1.ListNode{}
		// 获取出链表的最后一个元素
		for head.Next != nil {
			// 倒数第二个节点
			left = head
			// 能获取出来最后一个节点
			head = head.Next
		}
		// 倒数第一个节点设置为nil
		head.Next = prev
		// 将开始的最后一个节点设置为nil
		left.Next = nil
	}
	return head
}

// 排列序列
func getPermutation(n int, k int) string {
	//	 采用深度优先遍历，先排列出所有的组合，然后选择组合里面的数据
	//	 创建数组判断当前节点是否可以进行继续比较
	user := make([]bool, n)
	//	每一个数据
	pathNum := []int{}
	count := 0
	// 获取所有数据
	str := ""
	dfs_getPermutation(user, pathNum, n, &str, k, &count)
	return str
}

// user 最上面的一个节点是否可以选取
// 每次数组的节点是否满足
// 需要的参数数量
// 最后的数组
func dfs_getPermutation(user []bool, pathNum []int, n int, str *string, k int, count *int) {
	if len(pathNum) == n {
		*count++
		if *count == k {
			for _, value := range pathNum {
				*str += strconv.Itoa(value)
			}
		}
		return
	}
	// 遍历多少个数
	for i := 0; i < n; i++ {
		// 因为true的节点会被变为true，所以只能走其他的节点
		// 设置为true表示只能走其他的选择，
		if user[i] != true {
			user[i] = true
			pathNum = append(pathNum, i+1)
			dfs_getPermutation(user, pathNum, n, str, k, count)
			pathNum = pathNum[:len(pathNum)-1]
			user[i] = false
		}
	}
}

// 螺旋矩阵2
func generateMatrix(n int) [][]int {
	arr := make([][]int, n)

	for i := 0; i < n; i++ {
		arr[i] = make([]int, n)
	}
	// 因为螺旋矩阵，是n*n，
	// 所以开始的列是0 结束的列是n-1 开始的行是0,结束的行是n-1
	list_left := 0
	list_right := n - 1
	// 列
	row_left := 0
	row_right := n - 1
	index := 1
	for row_left <= row_right && list_left <= list_right {
		//	 先填行
		for k := row_left; k <= row_right; k++ {
			arr[list_left][k] = index
			index++
		}
		list_left++
		//	填写左边的列
		for k := list_left; k <= list_right; k++ {
			arr[k][row_right] = index
			index++
		}
		row_right--
		//if list_left < list_right && row_left < row_right {
		// 填写最下面的行
		for k := row_right; k >= row_left; k-- {
			arr[list_right][k] = index
			index++
		}
		list_right--
		//	填写左边的列
		for k := list_right; k >= list_left; k-- {
			arr[k][row_left] = index
			index++
		}
		row_left++
	}
	return arr
}

// 获取最后一个单词的长度
func lengthOfLastWord(s string) int {
	index := 0
	//flag  表示是否开始计算
	flag := false
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == 32 && !flag {
			continue
		}
		if s[i] != 32 {
			flag = true
			index++
		}
		if s[i] == 32 && flag {
			return index
		}
	}
	return index
}

// 获取最后一个单词的长度1
func lengthOfLastWord1(s string) int {
	space := strings.TrimSpace(s)
	arr := strings.Split(space, " ")
	return len(arr[len(arr)-1])
}

// 插入区间
func insert(intervals [][]int, newInterval []int) [][]int {
	left, right := newInterval[0], newInterval[1]
	//	 标记是否已经添加
	flag := false
	ans := [][]int{}
	for _, interval := range intervals {
		//	 判断插入那边 当前数组的第一个数大于插入的数，所以不会可以将数据直接添加
		if interval[0] > right {
			// 如果开始插入的数据没有插入，就先将数据插入
			if !flag {
				ans = append(ans, []int{left, right})
				flag = true
			}
			ans = append(ans, interval)
		} else if interval[1] < left { // 如果当前数组的第二个元素，小于插入元素的第一个元素，则可以直接添加
			ans = append(ans, interval)
		} else { // 当两个都不符合的时候，直接代表插入的数据可以选择，然后就对比数据
			left = min(left, interval[0])
			right = max(right, interval[1])
		}
	}
	if !flag {
		ans = append(ans, []int{left, right})
	}
	return ans
}

// 合并区间
func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	arr := [][]int{}
	prev := intervals[0]
	for i := 1; i < len(intervals); i++ {
		cue := intervals[i]
		//	将参数对比
		// 如果上一个数组的第二个参数，小于当前的参数，代表上一个数组可以直接添加进数组
		fmt.Println(prev, cue)
		if prev[1] < cue[0] {
			arr = append(arr, prev)
			// 原始的匹配数组已经成功，所以进行下一个数组的匹配
			prev = cue
		} else { //	如果上一个数组的第二个参数大于当前的参数，则进行替换
			prev[1] = max(prev[1], cue[1])
		}
	}
	arr = append(arr, prev)
	return arr
}

// 跳跃游戏
func canJump(nums []int) bool {
	//	 因为数组总的每个元素代表在该位置跳跃的最大长度
	//	i+num[i]表示当前位置跳跃的长度，和他的上一个位置的元素相比，谁远选择谁
	//	dp思想
	if nums == nil {
		return false
	}
	if len(nums) == 1 {
		return true
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	for i := 1; i < len(nums); i++ {
		if dp[i-1] >= i {
			dp[i] = max(dp[i-1], i+nums[i])
		}

		// 当大于或者等于就算到达了当前的目标
		if dp[i] >= len(nums)-1 {
			return true
		}
	}
	return false
}

// 跳跃游戏
func canJump1(nums []int) bool {
	if nums == nil {
		return false
	}
	//	 前面K个元素能跳的最远距离
	k := 0
	for i := 0; i < len(nums); i++ {
		if i > k {
			return false
		}
		//	 前面能跳的最远的距离
		temp := i + nums[i]
		//	 更新最远Julio
		k = max(k, temp)
	}
	//最远距离k不再改变,且没有到末尾元素
	return true
}

// 螺旋矩阵
func spiralOrder(matrix [][]int) []int {
	// 限制条件
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return []int{}
	}
	//	 因为是旋转型的 ，现在最外围的，然后第二层，直到碰到所指定的界限
	// 从第0列开始
	row_left := 0
	// 最后一列
	row_right := len(matrix[0]) - 1
	//	 第一行
	list_left := 0
	// 最后一行
	list_right := len(matrix) - 1
	arr := []int{}
	// 边界条件
	for row_left <= row_right && list_left <= list_right {
		// 首先计算最上面的行
		for i := row_left; i <= row_right; i++ {
			arr = append(arr, matrix[list_left][i])
		}
		//	 计算最右边的列
		for i := list_left + 1; i <= list_right; i++ {
			arr = append(arr, matrix[i][row_right])
		}
		// 当左边的列小于右边的列，就只计算从上向下的行 当上边的行大于下面的行，只计算从左到右的行
		if row_left < row_right && list_left < list_right {
			//	 计算最下方的行
			for i := row_right - 1; i > row_left; i-- {
				arr = append(arr, matrix[list_right][i])
			}
			//	计算最左边的行
			for i := list_right; i > list_left; i-- {
				arr = append(arr, matrix[i][row_left])
			}
		}

		row_left++
		row_right--
		list_left++
		list_right--
	}
	return arr
}

// N皇后2
func totalNQueens(n int) int {
	// 创建二维数组
	board := make([][]string, n)
	for i := 0; i < n; i++ {
		board[i] = make([]string, n)
	}
	//	 初始化所有的数据
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			board[i][j] = "."
		}
	}
	count := 0
	helper1(board, &count, 0, n)
	return count
}

func helper1(board [][]string, count *int, row, n int) {
	// 终止条件，当选择的数，等于给定的行，就将数据加入二维数组中
	if row == n {
		*count++
		return
	}
	for i := 0; i < n; i++ {
		// N皇后判断是否可以添加
		if isValid1(row, i, board) { // 当符合条件
			board[row][i] = "Q"
			helper1(board, count, row+1, n)
			board[row][i] = "."
		}

	}
}

// N皇后
func solveNQueens(n int) [][]string {
	res := [][]string{}
	// 创建二维数组
	board := make([][]string, n)
	for i := 0; i < n; i++ {
		board[i] = make([]string, n)
	}
	//	 初始化所有的数据
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			board[i][j] = "."
		}
	}
	helper(board, &res, 0, n)
	return res
}
func helper(board [][]string, res *[][]string, row, n int) {
	// 终止条件，当选择的数，等于给定的行，就将数据加入二维数组中
	if row == n {
		temp := make([]string, n)
		for i := 0; i < n; i++ {
			temp[i] = strings.Join(board[i], "")
		}
		*res = append(*res, temp)
		return
	}
	for i := 0; i < n; i++ {
		// N皇后判断是否可以添加
		if isValid1(row, i, board) { // 当符合条件
			board[row][i] = "Q"
			helper(board, res, row+1, n)
			board[row][i] = "."
		}
	}
}

// N皇后判断是否可以添加
func isValid1(row, col int, board [][]string) bool {
	n := len(board)
	for i := 0; i < row; i++ {
		if board[i][col] == "Q" {
			return false
		}
	}
	for i := 0; i < n; i++ {
		if board[row][i] == "Q" {
			return false
		}
	}

	for i, j := row, col; i >= 0 && j >= 0; i, j = i-1, j-1 {
		if board[i][j] == "Q" {
			return false
		}
	}
	for i, j := row, col; i >= 0 && j < n; i, j = i-1, j+1 {
		if board[i][j] == "Q" {
			return false
		}
	}
	return true

}

func maxSubArray2(nums []int) int {
	result := math.MinInt64
	len := len(nums)
	dp := make([]int, len)
	dp[0] = nums[0]
	result = dp[0]
	// 动态规划，上一个的值和当前值作比较，如果上一个值加上当前值比当前值大，则替换，如果小则不变
	for i := 1; i < len; i++ {
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		result = max(result, dp[i])
		fmt.Println(dp)
	}
	return result
}

// 最大子序和
func maxSubArray1(nums []int) int {

	len := len(nums)
	nums_max := math.MinInt64
	for i := 0; i < len; i++ {
		sum := 0
		for j := i; j < len; j++ {
			sum += nums[j]
			if sum > nums_max {
				nums_max = sum
			}
		}
	}
	return nums_max
}

func quickMul(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	// 分治算法，每次除以2
	y := quickMul(x, n/2)
	if n%2 == 0 {
		return y * y
	}
	return y * y * x
}

// 字母异味词分组
func groupAnagrams(strs []string) [][]string {
	arr_str := make([][]string, 0)
	lenght := len(strs)
	// 边界条件
	if lenght == 0 {
		return arr_str
	}
	if lenght == 1 {
		arr_str = append(arr_str, strs)
		return arr_str
	}
	//	 采用hash计算，key为排序后的数据，value为值
	map_str := make(map[string][]string)

	for _, value := range strs {
		int_arr := []int{}
		for _, val := range value {
			int_arr = append(int_arr, int(val))
		}
		sort.Ints(int_arr)
		str := ""
		for _, int_val := range int_arr {
			str += string(int_val)
		}
		// 当前值存在
		if map_str_val, ok := map_str[str]; ok {
			map_str_val = append(map_str_val, value)
			map_str[str] = map_str_val

		} else {
			map_str[str] = []string{value}
		}
	}

	for _, value := range map_str {
		arr_str = append(arr_str, value)
	}
	return arr_str
}

func rotate2(matrix [][]int) {
	lenght := len(matrix)
	if lenght == 0 {
		return
	}
	// 先将对着交换
	left, right := 0, lenght-1
	for left < right {
		matrix[left], matrix[right] = matrix[right], matrix[left]
		left++
		right--
	}
	//	 然后对角交换
	for i := 0; i < lenght; i++ {
		for j := i + 1; j < lenght; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
	fmt.Println(matrix)

}

// 旋转图像
func rotate1(matrix [][]int) {
	//	 旋转图像会发现规律，初始化每个二维数组的第一个元素会成为旋转图像之后的数组的最后一个元素
	//	例如：初始化每个二维数组的每个数组的第一个元素，抽取出来，按照倒序排列会成为，一次添加成为新的二维数组的第一个数组
	//	 获取二维数组的长度

	lenght := len(matrix)
	if lenght == 0 {
		return
	}
	//	新建一个新的二维数组
	arr := make([][]int, 0)
	// 单独获取一个数组的长度
	matrix_len := len(matrix[0])
	// 循环的是每个数组的个数
	for i := 0; i < matrix_len; i++ {
		//	 每次新建一个数组
		array := []int{}
		//	对二维数组进行循环
		for j := lenght - 1; j >= 0; j-- {
			// 获取每个数组的第一个数
			//	 获取当前的数组
			array = append(array, matrix[j][i])
		}
		//	 m
		arr = append(arr, array)
	}
	copy(matrix, arr)
	fmt.Println(matrix)
}

// 零钱问题
func coinChange2(coins []int, amount int) int {
	//	 初始化总数需要的数据
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
	}
	// 最小的零钱数量
	// 将长度循环
	for i := 1; i <= amount; i++ {
		//	对零钱的数量进行循环
		for _, coin := range coins {
			if i-coin < 0 {
				continue
			}
			dp[i] = min(dp[i], dp[i-coin]+1)
		}
	}
	fmt.Println(dp)
	if dp[amount] == amount+1 {
		return -1
	} else {
		return dp[amount]
	}

}

func permute(nums []int) [][]int {
	sort.Ints(nums)
	user := make([]bool, len(nums))
	pathNum := make([]int, 0)
	arr = make([][]int, 0)
	backtrack(nums, pathNum, user)
	return arr
}

//	def backtrack(路径, 选择列表):
//		if 满足结束条件:
//			result.add(路径)
//			return
//
//		for 选择 in 选择列表:
//			做选择
//			backtrack(路径, 选择列表)
//			撤销选择
func backtrack(nums, pathNums []int, used []bool) {
	if len(nums) == len(pathNums) {
		tmp := make([]int, len(nums))
		copy(tmp, pathNums)
		arr = append(arr, tmp)
		return
	}
	for i := 0; i < len(nums); i++ {
		if used[i] == true {
			continue
		}
		// 剪枝，因为排序过，肯定相同才会进行剪枝，也就是第一个以后，
		//if i-1 >= 0 && nums[i-1] == nums[i] && !used[i-1] {
		//	continue
		//}
		used[i] = true
		pathNums = append(pathNums, nums[i])
		fmt.Println(pathNums)
		// 树的情况采用了先序遍历
		backtrack(nums, pathNums, used)
		pathNums = pathNums[:len(pathNums)-1]
		used[i] = false
	}

}

func jump(nums []int) int {
	ans := 0 // 跳跃的次数
	end := 0 // 下一次起跳对的位置
	maxPos := 0
	for i := 0; i < len(nums)-1; i++ {
		maxPos = max(nums[i]+i, maxPos)
		if i == end {
			end = maxPos
			ans++
		}

	}
	return ans
}

// 字符串相乘
func multiply(num1 string, num2 string) string {
	// 因为乘数的位数M，和被乘数的位数N 最大总位数为M+N
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	// 创建一个总的数组存放数据
	l1, l2 := len(num1), len(num2)
	res := make([]int, l1+l2)
	for i := l1 - 1; i >= 0; i-- {
		n1 := num1[i] - '0'
		for j := l2 - 1; j >= 0; j-- {
			n2 := num2[j] - '0'
			//	 因为两个数相乘 会占据 i+j  和i+j+1的位置
			sum := (res[i+j+1] + int(n1*n2))
			// 将十位上的数赋值
			res[i+j+1] = sum % 10
			// 个位+=
			res[i+j] += sum / 10
		}
	}
	sum := ""
	for i := 0; i < len(res); i++ {
		if i == 0 && res[i] == 0 {
			continue
		}
		itoa := strconv.Itoa(int(res[i]))
		sum += itoa
	}
	return sum

}

// 接雨水
func trap(height []int) int {
	//	 总的数量
	sum := 0
	// 动态规划
	max_left := make([]int, len(height))
	max_right := make([]int, len(height))
	// 单独求出当前i左边最大的数
	for i := 1; i < len(height)-1; i++ {
		max_left[i] = max(max_left[i-1], height[i-1])
	}
	// 淡出求出i右边最大的数
	for i := len(height) - 2; i >= 0; i-- {
		max_right[i] = max(height[i+1], max_right[i+1])
	}
	for i := 1; i < len(height)-1; i++ {
		h_min := min(max_right[i], max_left[i])
		if h_min > height[i] {
			sum += h_min - height[i]
		}
	}

	// 按列求
	//h_min := 0
	//// 第一列最后一列不用考虑
	//for i := 1; i < len(height)-1; i++ {
	//	// 左边最高
	//	left_max := 0
	//	for j := i - 1; j >= 0; j-- {
	//		if height[j] > left_max {
	//			left_max = height[j]
	//		}
	//	}
	//	//	 右边最高
	//	right_max := 0
	//	for j := i + 1; j < len(height); j++ {
	//		if height[j] > right_max {
	//			right_max = height[j]
	//		}
	//	}
	//	//	 找出两端较小的数
	//	h_min = min(left_max, right_max)
	//
	//	// 左右最大较小的一个大于当前的数，才会增加水的数量
	//	if h_min > height[i] {
	//		sum += h_min - height[i]
	//	}
	//
	//}

	// 按行求
	//nummax := 0
	//// 求出最大的高度
	//for i := 0; i < len(height); i++ {
	//	if nummax <= height[i] {
	//		nummax = height[i]
	//	}
	//}
	//for i := 1; i <= nummax; i++ {
	//	//	 标记是否开始更新
	//	flag := false
	//	temp := 0
	//	for j := 0; j < len(height); j++ {
	//		//	当进行更新的时候并且当前高度小于雨水的高度才会进行更新
	//		if flag && height[j] < i {
	//			//	当前雨水高度一直小于指定的高度，就会一直加
	//			temp++
	//		}
	//		// 当前的高度大于或者等于雨水的高度，初始化tmp
	//		if i <= height[j] {
	//			sum += temp
	//			temp = 0
	//			flag = true
	//		}
	//	}
	//}
	return sum
}

// 缺失的第一个整数
func firstMissingPositive(nums []int) int {
	sort.Ints(nums)
	re := 1
	for i := 0; i < len(nums); i++ {
		if nums[i] == re {
			re++
		}

	}
	return re

	//m := make(map[int]int)
	//for key, value := range nums {
	//	m[value] = key
	//}
	//index := 1
	//for i := 1; i <= len(nums)+1; i++ {
	//	if _, ok := m[i]; !ok {
	//		index = i
	//		return index
	//	}
	//}
	//return index+1
}

// 总数之和2
func combinationSum2(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	res := [][]int{}
	dfs2(candidates, nil, target, 0, &res)
	return res
}

func dfs2(candidates, nums []int, target, left int, res *[][]int) {
	// 等于0 就直接结算
	if target == 0 {
		tmp := make([]int, len(nums))
		copy(tmp, nums)
		*res = append(*res, tmp)
		return
	}
	for i := left; i < len(candidates); i++ { // left限定，形成分支
		if i != left && candidates[i] == candidates[i-1] { // *同层节点 数值相同，剪枝
			continue
		}
		if target < candidates[i] {
			return
		}
		dfs2(candidates, append(nums, candidates[i]), target-candidates[i], i+1, res) //*分支 i+1避免重复
	}

}

// 总数之和1
func combinationSum(candidates []int, target int) [][]int {
	// 将数组排序
	sort.Ints(candidates)
	// 清理数组
	result = result[0:0]
	dfs(target, candidates, []int(nil))
	return result
}

// 深度算法优先
func dfs(sum int, candidates []int, path []int) {
	//	 终止条件
	if sum == 0 {
		result = append(result, append([]int(nil), path...))
	}
	//	 遍历点前层中所有的元素
	for i, num := range candidates {
		if sum-num < 0 {
			return
		}
		dfs(sum-num, candidates[i:], append(path, num))
	}
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
