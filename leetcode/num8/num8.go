package num8

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

// 441. 排列硬币
func arrangeCoins(n int) int {

	return 0
}

// 440. 字典序的第K小数字
func findKthNumber(n, k int) int {
	count := 1
	prefix := 1
	for count < k {
		cnt := getCount(prefix, n)
		// 说明不在当前的前缀上
		if count+cnt <= k {
			// 那就试试下一个前缀
			prefix++
			count += cnt
		} else {
			// 在该前缀上，下去一层
			prefix *= 10
			count++
		}
	}
	return prefix
}

func getCount(prefix, n int) int {
	next := prefix + 1
	count := 0
	for prefix <= n {
		//比如n是195的情况195到100有96个数
		count += min(n+1, next) - prefix
		prefix *= 10
		next *= 10
	}
	return count
}

// 386. 字典序排数
func lexicalOrder(n int) []int {
	res := []int{}
	// 因为是1-9的开始，所以最开始的初始遍历是1-9
	for i := 1; i <= 9; i++ {
		lexicalOrderDfs(i, n, &res)
	}
	fmt.Println(res)
	return res
}

// 每一次进入递归的数据，都是前一个数*10 + 循环的数据
func lexicalOrderDfs(x, n int, arr *[]int) {
	if x > n {
		return
	}
	*arr = append(*arr, x)
	x = x * 10
	for i := 0; i <= 9; i++ {
		lexicalOrderDfs(x+i, n, arr)
	}
}

// 438. 找到字符串中所有字母异位词
func findAnagrams(s string, p string) []int {
	arr := []int{}
	PArr := [26]int{}
	for _, val := range p {
		PArr[val-'a']++
	}
	SArr := [26]int{}

	for key, val := range s {
		SArr[val-'a']++
		if key >= len(p) {
			SArr[s[key-len(p)]-'a']--
		}
		if SArr == PArr {
			arr = append(arr, key-len(p)+1)
		}
	}
	return arr
}

// 437. 路径总和 III
func pathSum(root *TreeNode, targetSum int) int {
	if root == nil {
		return 0
	}
	//	 将数据的值相加，分别是当前节点 左子节点  又子节点
	return pathSumNode(root, targetSum) + pathSum(root.Left, targetSum) + pathSum(root.Right, targetSum)
}

func pathSumNode(root *TreeNode, targetSum int) int {
	if root == nil {
		return 0
	}
	// 开始0条路径
	cnt := 0
	//	 当targetSum 等于当前节点的值，表示会新增一条路径
	if targetSum == root.Val {
		cnt++
	}
	//	 进入下一个节点进行判断，进入之前应该减去当前节点的值
	cnt += pathSumNode(root.Left, targetSum-root.Val)
	cnt += pathSumNode(root.Right, targetSum-root.Val)
	return cnt
}

// 436. 寻找右区间
func findRightInterval(intervals [][]int) []int {
	r := []int{}
	for i := 0; i < len(intervals); i++ {
		min, minIdx := math.MaxInt64, -1
		for j := 0; j < len(intervals); j++ {
			if i == j {
				continue
			}
			if intervals[i][1] <= intervals[j][0] {
				if intervals[j][0] < min {
					min = intervals[j][0]
					minIdx = j
				}
			}
		}
		if intervals[i][1] == intervals[i][0] {
			minIdx = 0
		}
		r = append(r, minIdx)
	}
	return r
}

// 435. 无重叠区间
func eraseOverlapIntervals(intervals [][]int) int {
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] == intervals[j][0] {
			return intervals[i][1] < intervals[j][1]
		} else {
			return intervals[i][0] < intervals[j][0]
		}
	})
	num := 0
	//  终止的位置
	end := intervals[0][1]
	// 总结就是，因为要获取最小删除的数，所以排序过后，右边界越小越号，因为容易越小代表需要的数组越多，删除的就越少
	for i := 1; i < len(intervals); i++ {
		//	 先判断数组第一个数是否小于
		if end <= intervals[i][0] {
			// 因为原来的数组小于当前数组的第一个数，所以重新开始的数因为为当前数组的最后一个数
			end = intervals[i][1]
		} else {
			// 如果原来的字符大于当前数组的第一个字符，因为要计算的是最小的删除的数量，所以新的开始位置选择当前数组的第二个字符
			if end > intervals[i][1] {
				end = intervals[i][1]
			}
			num++
		}
	}
	return num
}

// 434. 字符串中的单词数
func countSegments(s string) int {
	res := 0
	for key, val := range s {
		if val != ' ' && (key == 0 || s[key-1] == ' ') {
			res++
		}
	}
	return res
}

// 433. 最小基因变化
func minMutation(start string, end string, bank []string) int {
	//	 将bank里面的数据添加到map中进行做判断
	bankMap := make(map[string]bool)
	for _, val := range bank {
		// true表示没有经历过变化
		bankMap[val] = true
	}
	strArr := []string{"A", "C", "G", "T"}
	//	 次数
	num := 0
	queue := []string{start}
	// 采用bfs进行计算
	for len(queue) != 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			word := queue[0]
			queue = queue[1:]
			if word == end {
				return num + 1
			}
			for c := 0; c < len(word); c++ {
				for j := 0; j < len(strArr); j++ {
					newword := word[:c] + strArr[j] + word[c+1:]
					if bankMap[newword] == true {
						queue = append(queue, newword)
						bankMap[newword] = false
					}
				}
			}
		}
		num++
	}
	return -1
}

// 432. 全 O(1) 的数据结构
type AllOne struct {
	// 所有为1的数据
	One map[string]int
	//	所有数量不为1 的数据
	NotOne map[string]int
}

func Constructor_432() AllOne {
	return AllOne{
		One:    map[string]int{},
		NotOne: map[string]int{},
	}
}

func (this *AllOne) Inc(key string) {
	_, ok1 := this.One[key]
	_, ok2 := this.NotOne[key]
	if !ok1 && !ok2 {
		this.One[key]++

	} else if !ok2 {
		this.NotOne[key] = 2
		delete(this.One, key)
	} else if !ok1 {
		this.NotOne[key]++
	}
}

func (this *AllOne) Dec(key string) {
	_, ok1 := this.One[key]
	_, ok2 := this.NotOne[key]
	if !ok2 && !ok1 {
		return
	} else if !ok2 {
		delete(this.One, key)
	} else if !ok1 {
		this.NotOne[key]--
		if this.NotOne[key] == 1 {
			delete(this.NotOne, key)
			this.One[key]++
		}
	}
}

func (this *AllOne) GetMaxKey() string {
	if len(this.One) == 0 && len(this.NotOne) == 0 {
		return ""
	}
	num := 0
	str := ""
	if len(this.NotOne) != 0 {
		for key, val := range this.NotOne {
			if val > num {
				num = val
				str = key
			}
		}
		return str
	}
	for key, _ := range this.One {
		return key
	}
	return ""
}

func (this *AllOne) GetMinKey() string {
	if len(this.One) == 0 && len(this.NotOne) == 0 {
		return ""
	}
	if len(this.One) != 0 {
		for key, _ := range this.One {
			return key
		}
	}
	num := math.MaxInt64
	str := ""
	for key, val := range this.NotOne {
		if num > val {
			num = val
			str = key
		}
	}
	return str
}

//430. 扁平化多级双向链表
func flatten(root *Node) *Node {
	if root == nil {
		return nil
	}
	// 因为是双向链表，每次用的到的数据都是 next指针和child指针
	slack := []*Node{root}
	next := &Node{}
	per := next
	for len(slack) != 0 {
		cur := slack[len(slack)-1]
		slack = slack[:len(slack)-1]
		per.Next = cur
		cur.Prev = per
		per.Child = nil
		if cur.Next != nil {
			slack = append(slack, cur.Next)
		}
		if cur.Child != nil {
			slack = append(slack, cur.Child)
		}
		per = per.Next
	}
	next = next.Next
	next.Prev = nil
	return next
}

// 429. N 叉树的层序遍历
func levelOrder(root *Node) [][]int {
	if root == nil {
		return nil
	}
	//	 采用宽度优先遍历
	queue := []*Node{root}
	arr := [][]int{}

	for len(queue) != 0 {
		// 存储下一层的数据
		next := []*Node{}
		list := []int{}
		// 一次性将数组里面的所有数据取出来
		for len(queue) != 0 {
			cur := queue[0]
			queue = queue[1:]
			list = append(list, cur.Val)
			next = append(next, cur.Child)
		}

		queue = append(queue, next...)
		arr = append(arr, list)
	}
	return arr
}

// 424. 替换后的最长重复字符
func characterReplacement(s string, k int) int {
	//	 采用滑动窗口实现
	//	 创建一个byte类型，26个字符一对对应
	//	 每移动一次的话，判断字符符合要求， 当前窗口内的字符总长度-最大字符长度=other其他字符出现的次数<k
	//	 如果不满足的话 左边的移动，并且bute数组里面对应的数量--
	cnt := make([]int, 26)
	var check func() bool
	// 判断当前字符 l-r之间是否符合条件
	// 这个区间的总长度-最大字符串的长度 =other<k
	check = func() bool {
		per, sum := 0, 0
		for _, val := range cnt {
			sum += val
			per = max(per, val)
		}
		return sum-per <= k
	}
	ans := 0
	l, r := 0, 0
	for ; r < len(s); r++ {
		// 将每次循环的字符 如果出现次数加一
		cur := s[r] - 'A'
		cnt[cur]++
		// 判断是否符合要求
		for !check() {
			// 不符合的话 数据减一 l++
			del := s[l] - 'A'
			cnt[del]--
			l++
		}
		// 总长度为r-l+1 比如 0-1之间 但是是两个数
		ans = max(ans, r-l+1)
	}
	return ans
}

// 423. 从英文中重建数字
func originalDigits(s string) string {
	tmp := make(map[rune]int, len(s))
	for _, i := range s {
		tmp[i]++
	}
	//  z w x g u 室友0,2,4,6,8 中出现

	count := [10]int{}
	count[0] = tmp['z']
	count[2] = tmp['w']
	count[4] = tmp['u']
	count[6] = tmp['x']
	count[8] = tmp['g']
	// h在3和8中出现 所以h-8的个数就为3的个数
	count[3] = tmp['h'] - count[8]
	count[1] = tmp['o'] - count[0] - count[2] - count[4]
	count[5] = tmp['f'] - count[4]
	count[7] = tmp['s'] - count[6]
	count[9] = tmp['i'] - count[5] - count[8] - count[6]
	var res []byte
	for p, i := range count {
		res = append(res, bytes.Repeat([]byte{byte('0' + p)}, i)...)
	}
	return string(res)
}

// 419. 甲板上的战舰
func countBattleships(board [][]byte) int {
	//	 返回的总数量
	count := 0
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			// 如果军舰是单独的一个，
			// 1. 是第0行
			// 不是第0行的时候 上面没有军舰
			// 是第一列，不是第一列的时候，左边没有军舰
			if board[i][j] == 'X' && (i == 0 || board[i-1][j] == '.') && (j == 0 || board[i][j-1] == '.') {
				count++
			}
		}
	}
	return count
}

//417. 太平洋大西洋水流问题
func pacificAtlantic(heights [][]int) [][]int {
	//	 因为获取的是那些水可以流向太平洋 同事也能流到太平洋
	//	先分别设置两个可以流动的数据，然后再汇总
	//	 返回结果的数组
	res := make([][]int, 0)
	//	 多少行 多少列
	m, n := len(heights), len(heights[0])
	//  标记能够达到太平洋的下标
	P := make([][]bool, m)
	// 标记能够达到大西洋的下标
	A := make([][]bool, m)
	// 初始化
	for i := 0; i < m; i++ {
		P[i] = make([]bool, n)
		A[i] = make([]bool, n)
	}
	var pacificAtlanticDfs func(visited [][]bool, startx, starty int, numSize int)
	pacificAtlanticDfs = func(visited [][]bool, startx, starty int, numSize int) {
		//	 出边界的话 就返回
		//	 已经访问过了返回
		//	 不能向低处流 返回
		if notInArea(startx, starty, m, n) || visited[startx][starty] || heights[startx][starty] < numSize {
			return
		}

		visited[startx][starty] = true
		if P[startx][starty] == A[startx][starty] {
			res = append(res, []int{startx, starty})
		}
		pacificAtlanticDfs(visited, startx+1, starty, heights[startx][starty])
		pacificAtlanticDfs(visited, startx-1, starty, heights[startx][starty])
		pacificAtlanticDfs(visited, startx, starty+1, heights[startx][starty])
		pacificAtlanticDfs(visited, startx, starty-1, heights[startx][starty])
	}
	// 上太平洋 下大西洋
	for i := 0; i < n; i++ {
		pacificAtlanticDfs(P, 0, i, math.MinInt64)
		pacificAtlanticDfs(A, m-1, i, math.MinInt64)
	}
	//	 左太平洋，右大西洋
	for i := 0; i < m; i++ {
		pacificAtlanticDfs(P, i, 0, math.MinInt64)
		pacificAtlanticDfs(A, i, n-1, math.MinInt64)
	}
	return res
}

// 判断是否在边界之外
func notInArea(x, y, m, n int) bool {
	return x < 0 || y < 0 || x >= m || y >= n
}

//416. 分割等和子集
func canPartition(nums []int) bool {
	if len(nums) <= 1 {
		return false
	}
	// 对于每一个都是选择或者不选择
	// 计算所有的数之和，如果为奇数则一定不满足条件
	sum := 0
	for _, val := range nums {
		sum += val
	}
	if sum%2 != 0 {
		return false
	}
	// 二维数组的bp 长宽 限制为 i,avg
	avg := sum / 2
	dp := make([]bool, avg+1)
	dp[0] = true
	for i := 0; i < len(nums); i++ {
		for j := avg; j >= nums[i]; j-- {
			dp[j] = dp[j] || dp[j-nums[i]]
		}
	}
	return dp[avg]
	// dfs 计算
	//m := make(map[int]bool)
	//if sum%2 != 0 {
	//	return false
	//}
	//dfs := canPartition_dfs(sum/2, 0, 0, nums, m)
	//return dfs
}

func canPartition_dfs(sum, i, avg int, nums []int, m map[int]bool) bool {
	// base case
	// 长度到了leng的时候 或者求的和已经爆出了的情况 返回flace\
	key := avg*1000 + i
	val, ok := m[key]
	if ok {
		return val
	}
	if i == len(nums) || avg > sum {
		m[key] = false
		return m[key]
	}
	if sum == avg {
		m[key] = true
		return m[key]
	}
	// 每次选择或者不选

	m[key] = canPartition_dfs(sum, i+1, avg+nums[i], nums, m) || canPartition_dfs(sum, i+1, avg, nums, m)
	return m[key]
}

// 415. 字符串相加
func addStrings(num1 string, num2 string) string {
	nb1, nb2 := len(num1), len(num2)
	// 将两个字符变为一样的长度
	if nb1 > nb2 {
		for i := 0; i < nb1-nb2; i++ {
			num2 = "0" + num2
		}
	} else {
		for i := 0; i < nb2-nb1; i++ {
			num1 = "0" + num1
		}
	}
	str := ""
	num := 0
	nb1, nb2 = len(num1), len(num2)
	for i := nb1 - 1; i >= 0; i-- {
		//		 每次提前两个数
		atoi1, _ := strconv.Atoi(string(num1[i]))
		atoi2, _ := strconv.Atoi(string(num2[i]))
		sum := num + atoi1 + atoi2
		if sum >= 10 {
			str = strconv.Itoa(sum-10) + str
			num = 1
		} else {
			num = 0
			str = strconv.Itoa(sum) + str
		}
	}

	if num > 0 {
		str = strconv.Itoa(num) + str
	}
	return str
}

// 414. 第三大的数
func thirdMax(nums []int) int {
	m := make(map[int]int)
	for _, val := range nums {
		m[val] = val
	}
	arr := []int{}
	for _, val := range m {
		arr = append(arr, val)
	}
	sort.Ints(arr)
	if len(arr) <= 2 {
		return arr[len(arr)-1]
	}
	return arr[len(arr)-3]
}

// 413. 等差数列划分
func numberOfArithmeticSlices(nums []int) int {
	dp := make([]int, len(nums))
	if len(nums) <= 2 {
		return 0
	}
	for i := 2; i < len(nums); i++ {
		// 连续等差数列就是 当前数减去上一个数 等于上一个数减去上上一个数
		if nums[i]-nums[i-1] == nums[i-1]-nums[i-2] {
			// 如果是等差数列就将上次的等差数列+1
			dp[i] = dp[i-1] + 1
		}
	}
	sum := 0
	// 最后将所有的等差数列汇总
	for _, val := range dp {
		sum += val
	}
	return sum
}

// 412. Fizz Buzz
func fizzBuzz(n int) []string {
	str := []string{}
	for i := 1; i <= n; i++ {
		if i%15 == 0 {
			str = append(str, "FizzBuzz")
		} else if i%5 == 0 {
			str = append(str, "Buzz")
		} else if i%3 == 0 {
			str = append(str, "Fizz")
		} else {
			str = append(str, strconv.Itoa(i))
		}
	}
	return str
}

// 410. 分割数组的最大值
func splitArray(nums []int, m int) int {
	//	因为最终的答案只会存在数组中的最大值至数组中的和之间
	//	因为是计算两个数中间的数，那么就可以采用二分查找算法
	left, right := 0, 0
	for _, val := range nums {
		//	 初始化两个数据
		right += val
		if val > left {
			left = val
		}
	}
	//	 二分查找的条件
	for left < right {
		mid := left + (right-left)/2
		// 表示符合要求的数组 并且每个数组的和都小于mid ，但是数量比m多，表示mid太小，那么mid就应该向右移动
		if split(nums, mid, m) > m {
			left = mid + 1
		} else if split(nums, mid, m) <= m {
			// 表示符合要求的数组，并且每个数组的和都小于mid，但是数量不足m，那么就表示mid太大，需要向左移动
			right = mid
		}
	}
	return left
}
func split(nums []int, max, m int) int {
	// 因为count默认都是会有一个数组
	sum, count := 0, 1
	for _, val := range nums {
		if sum+val <= max {
			sum = sum + val
		} else {
			count++
			sum = val
		}

	}
	return count
}

// 409. 最长回文串
func longestPalindrome(s string) int {
	// 组装的话，就是判断前面所有的数据都为0，
	//	 因为1个字符也是回文 所以最低为1
	m_str := make(map[byte]int)
	sum := 0
	for i := 0; i < len(s); i++ {
		m_str[s[i]]++
	}
	for _, val := range m_str {
		if val >= 2 {
			if val%2 == 0 {
				sum += val
			} else {
				sum += val - 1
			}
		}
	}
	if sum != len(s) {
		sum++
	}
	return sum
}

// 407. 接雨水 II
func trapRainWater(heightMap [][]int) int {
	n, m := len(heightMap), len(heightMap[0])
	outBound := make([][]int, n)
	for i := 0; i < len(outBound); i++ {
		outBound[i] = make([]int, m)
	}
	sum := 0
	flag := true
	round := 1
	// 重复一次又一次的在四周取最小值
	for flag {
		flag = false
		// 计算内围的每个点最低外边：顺序初始化一遍
		for i := 1; i < n-1; i++ {
			for j := 1; j < m-1; j++ {
				// 上边和左边的最小值
				newBound := min(max(heightMap[i-1][j], outBound[i-1][j]), max(heightMap[i][j-1], outBound[i][j-1]))
				if newBound < outBound[i][j] || round == 1 {
					flag = true
					outBound[i][j] = newBound
				}
			}
		}
		//	 计算 右边和下边的最小值
		for i := n - 2; i > 0; i-- {
			for j := m - 2; j > 0; j-- {
				newBound := min(max(heightMap[i+1][j], outBound[i+1][j]), max(heightMap[i][j+1], outBound[i][j+1]))
				if newBound < outBound[i][j] || round == 1 {
					outBound[i][j] = newBound
					flag = true
				}
			}
		}
		round++
	}
	for i := 1; i < n-1; i++ {
		for j := 1; j < m-1; j++ {
			if heightMap[i][j] < outBound[i][j] {
				sum += outBound[i][j] - heightMap[i][j]
			}
		}
	}
	return sum
}

//406. 根据身高重建队列
func reconstructQueue(people [][]int) [][]int {
	sort.Slice(people, func(i, j int) bool {
		if people[i][0] == people[j][0] {
			return people[i][1] < people[j][1]
		} else {
			return people[i][0] > people[j][0]
		}
	})
	// 按照k值插入到index=k的地方，index之后的往后移动
	for i, p := range people {
		copy(people[p[1]+1:i+1], people[p[1]:i+1])
		people[p[1]] = p
	}
	return people
}

// 404. 左叶子之和
func sumOfLeftLeaves(root *TreeNode) int {
	if root == nil {
		return 0
	}
	num := 0
	sumOfLeftLeaves_left(root, &num)
	return num
}

func sumOfLeftLeaves_left(root *TreeNode, num *int) {
	if root == nil {
		return
	}
	if root.Left != nil && root.Left.Left == nil && root.Left.Right == nil {
		*num += root.Left.Val

	}
	sumOfLeftLeaves_left(root.Left, num)
	sumOfLeftLeaves_left(root.Right, num)
}

// 403. 青蛙过河
func canCross(stones []int) bool {
	if len(stones) == 1 {
		return true
	}
	//	 动态规划实现
	dp := make([][]bool, len(stones)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(stones)+1)
	}
	dp[0][0] = true
	for i := 1; i < len(stones); i++ {
		// stones[i]是从stones[i-1]跳过来，此时 k 最小 ，又因为到达 stones[i-1]的最大距离是从stones[1]跳到stones[i-1],每跳一步，能跳跃的最大的距离k最多只能增加1
		if stones[i]-stones[i-1] > i {
			return false
		}
	}
	// dp[i][k]代表 跳到第 i 个 位子时的跳跃距离是k dp[0][0]=true   dp[1][1]=true
	// 每跳一次，至少跳了1格，但是那一次跳跃的距离，至多只能增加1
	// 从第一块石头开始
	for i := 1; i < len(stones); i++ {
		for j := i - 1; j >= 0; j-- {
			k := stones[i] - stones[j]
			// 到达第j个，跳跃的最大距离也就最大是j，那么调到i时的距离k 最大也就是 j+1 不可能大于 j+1
			if k > j+1 {
				break
			}
			dp[i][k] = dp[j][k-1] || dp[j][k] || dp[j][k+1]
			if i == len(stones)-1 && dp[i][k] {
				return true
			}
		}

	}

	return false
	// 回溯算法实现
	//can_map := make(map[int]bool)
	//return canCross_dfs(stones, 0, 0, can_map)
}

// stones 表示原数组 k表示上次跳跃了k各单位
func canCross_dfs(stones []int, index, k int, can_map map[int]bool) bool {
	key := index*1000 + k
	if can_map[key] { // 当前计算是否出现过 出现过直接返回false
		// 因为第一次遇见 返回的是false 唯有false 才会继续搜索别的分支
		return false
	} else { // 当前结果是否出现过 ，没有出现就直接返回true
		can_map[key] = true
	}
	for i := index + 1; i < len(stones); i++ {
		// 判断两次进行的次数
		num := stones[i] - stones[index]
		if num <= k+1 && num >= k-1 {
			if canCross_dfs(stones, i, num, can_map) {
				return true
			}
		} else if num > k+1 {
			break
		}
	}
	return index == len(stones)-1
}

// 402. 移掉 K 位数字
func removeKdigits(num string, k int) string {
	//	 如果长度一样的话 直接返回0
	if len(num) == k {
		return "0"
	}
	stack := []int32{}
	for _, val := range num {
		//	判断当前的值是否需要将栈顶元素弹出来
		for len(stack) > 0 && stack[len(stack)-1] > val && k > 0 {
			stack = stack[:len(stack)-1]
			k--
		}
		// 判断数据是不是0
		if val != '0' || len(stack) != 0 {
			stack = append(stack, val)
		}
	}
	//	 如果还没有进行删除结束
	for len(stack) > 0 && k > 0 {
		stack = stack[:len(stack)-1]
		k--
	}
	if len(stack) == 0 {
		return "0"
	}
	return string(stack)
}

//401. 二进制手表
func readBinaryWatch(turnedOn int) []string {
	result := []string{}
	for i := 0; i < 12; i++ {
		for j := 0; j < 60; j++ {
			// 采用二进制形式表示
			b1 := fmt.Sprintf("%b", i)
			b2 := fmt.Sprintf("%b", j)
			// 二进制上面有多少1 表示
			sumOne := strings.Count(b1, "1") + strings.Count(b2, "1")
			fmt.Println(b1, b2)
			if sumOne == turnedOn {
				result = append(result, fmt.Sprintf("%d:%02d", i, j))
			}
		}
	}
	return result

}

// 400. 第 N 位数字
func findNthDigit(n int) int {

	// 找到n在第几组
	i, temp := 1, 1
	for ; n > i*temp*9; i++ {
		// 将去当前的区间范围的数
		// 1-9 =9
		//	10-99 = 90 因为是两位所以*2
		//	100-999 因为是三位所以*3
		n -= i * 9 * temp
		temp *= 10
	}
	if n%i == 0 {
		return (temp + n/i - 1) % 10
	}
	num := temp + n/i
	pos := i - n%i
	return (num % IntPow(10, pos+1)) / IntPow(10, pos)
}
func IntPow(a, b int) int {
	res := 1
	for b > 0 {
		res *= a
		b--
	}
	return res
}

// 399. 除法求值
// 判断两个点有没有连接 1， 无向图 2， 并查集
func calcEquation(equations [][]string, values []float64, queries [][]string) []float64 {
	res := make([]float64, 0)
	maps := make(map[string]*UfNode)
	for key, val := range equations {

		v1, ok1 := maps[val[0]]
		v2, ok2 := maps[val[1]]
		// 当两个数都不存在map中
		if !ok1 && !ok2 {
			//创建两个 并查集的树
			// 例如 a/b=3 那么 a=3 b=1
			p1, p2 := NewUfNode(values[key]), NewUfNode(1)
			maps[val[0]], maps[val[1]] = p1, p2
			// 组成树
			p1.parent = p2
		} else if !ok1 {
			//	 ok1 没有 但是ok2有
			p2 := findParent(v2)
			//	 v1 /v2 = k v1=v2*k
			p1 := NewUfNode(v2.value * values[key])
			maps[val[0]] = p1
			p1.parent = p2
		} else if !ok2 {
			//	 ok1有 ok2 没有
			p1 := findParent(v1)
			//v1 /v2=k v2 =v1/k
			p2 := NewUfNode(v1.value / values[key])
			maps[val[1]] = p2
			p2.parent = p1
		} else {
			// 数组里面的两个数都可以在map中找到的话 就可以进行并查集合并
			union(v1, v2, values[key], maps)
		}
	}
	//	 合并完并查集之后开始查找
	for _, val := range queries {
		v1, ok1 := maps[val[0]]
		v2, ok2 := maps[val[1]]
		//	 如果两个字符都出现在了同一个集合，不同集合无法计算
		if ok1 && ok2 && findParent(v1) == findParent(v2) {
			res = append(res, v1.value/v2.value)
		} else {
			res = append(res, -1.0)
		}
	}
	return res
}

// 合并两个树 为一棵树
func union(node1, node2 *UfNode, num float64, maps map[string]*UfNode) {
	p1, p2 := findParent(node1), findParent(node2)
	// 当两个点的更节点不同 才会组成同一个树
	if p1 != p2 {
		// 将两个树重新计算
		//把一颗子树挂到另一棵树的时候，要把挂上去的树乘以一个和父树的比率
		//这个比率要保证两棵树的所有节点的value相除都是正确的结果
		//比如A树中  a/b = 3  其中 a初始化为3，b初始化为1
		//B树中  c/d = 5 其中c为5，d为1
		//原本A树和B树是不相干的两个集合，此时有一个条件为 a/c = 5，a =5 c=1 那么就可以将A集合中的a和B集合中的c节点联系起来，两个集合最终就能连成一个集合
		//那么直接将A挂到B上来的话，a为3c为5  3/5很明显 != a/c != 5，因此要将A树挂到B树上的话，就要将A树整体扩大或者缩小到可以满足a/c=5的地步
		//那这个倍数怎么算？ a/c = 5   a=c*5，那么我只要将a变成c*5的值就行了,那a乘以多少=c*5呢，设为ratio,   a * ratio=c*5    =》 ratio = c*5/a
		//这里设node1为A树节点，node2为B树节点，将A树挂到B树
		ratio := node2.value * num / node1.value
		//将A树所有节点整体扩大，这样A树里面的所有除法结果依然不变，并且能兼容B树的数字
		for k, v := range maps {
			// 如果根节点一样，那么所有匹配到的猪 都需要乘以这个倍数
			if findParent(v) == p1 {
				maps[k].value *= ratio
			}
		}
		p1.parent = p2
	}
}

// 采用并查集实现
type UfNode struct {
	value  float64
	parent *UfNode
}

// 初始化node函数
func NewUfNode(value float64) *UfNode {
	node := &UfNode{value: value}
	// 套娃 自己关联自己本身
	node.parent = node
	return node
}

// 返回当前并查集的头结点
func findParent(node *UfNode) *UfNode {
	if node == node.parent {
		return node
	}
	// 将所有的上序节点都指向根节点
	node.parent = findParent(node.parent)
	return node.parent
}

//398. 随机数索引
type Solution_389 struct {
	Arr []int
}

func Constructor_398(nums []int) Solution_389 {
	return Solution_389{
		Arr: nums,
	}
}

func (this *Solution_389) Pick(target int) int {
	arr := []int{}
	for key, val := range this.Arr {
		if val == target {
			fmt.Println(key)
			arr = append(arr, key)
		}
	}
	fmt.Println(arr)
	return rand.Intn(arr[len(arr)])
}

//397. 整数替换
func integerReplacement(n int) int {
	int_map := map[int]int{}
	return integerReplacement_dfs(n, int_map)
}

func integerReplacement_dfs(n int, int_map map[int]int) int {
	if n == 1 {
		return 0
	}
	if val, ok := int_map[n]; ok {
		return val
	}
	if n%2 == 0 {
		// 如果为偶数的话 直接进行除2进行处理
		int_map[n] = 1 + integerReplacement_dfs(n/2, int_map)
	} else {
		// 如果为奇数 要计算+1 还是-1  然后取最小
		int_map[n] = 1 + min(integerReplacement_dfs(n+1, int_map), integerReplacement_dfs(n-1, int_map))
	}
	return int_map[n]
}

//396. 旋转函数
func maxRotateFunction(nums []int) int {
	lenght := len(nums)
	sum := math.MinInt64
	// 计算第一个位置
	cur := 0
	dp := make([]int, lenght)
	// 整个数组的和
	prt := 0
	for key, val := range nums {
		cur += key * val
		prt += val
	}
	dp[0] = cur
	start := lenght - 1
	for i := 1; i < len(dp); i++ {

		dp[i] = dp[i-1] - (lenght-1)*nums[start] + prt - nums[start]
		start--
	}
	for _, val := range dp {
		sum = max(val, sum)
	}

	return sum
}

//395. 至少有 K 个重复字符的最长子串
// 明天再做一次
func longestSubstring(s string, k int) int {
	// 最后的长度
	return longestSubstring_helper(0, len(s)-1, k, s)
}

func longestSubstring_helper(start, end, k int, s string) int {
	if end-start+1 < k {
		return 0
	}
	//	 统计当前区间出现的字符
	freq := make(map[byte]int, end-start+1)
	// 判断数据有多少重复
	for i := start; i <= end; i++ {
		freq[s[i]]++
	}
	// 合法剪枝，在当前区间上面，该字符的长度不满足k的时候进行移动指针
	for end-start+1 >= k && freq[s[start]] < k {
		start++
	}
	for end-start+1 >= k && freq[s[end]] < k {
		end--
	}
	if end-start+1 < k {
		return 0
	}
	for i := start; i <= end; i++ {
		if freq[s[i]] < k {
			return max(longestSubstring_helper(start, i-1, k, s), longestSubstring_helper(i+1, end, k, s))
		}
	}
	return end - start + 1
}

//394. 字符串解码
func decodeString(s string) string {
	str := ""
	// 采用栈的形式存放，因为不可能出现数字
	// 出现数字的可能就是 几倍中括号的数字
	str_stack := []string{}
	num_stack := []int{}
	// 当遇见数字的时候放进数字栈里面
	// 让遇见中括号的时候，将数字放进重复数字栈里面
	// 当遇见】时，两个出栈组装数据
	num := 0
	for _, char := range s {
		if char >= '0' && char <= '9' {
			atoi, _ := strconv.Atoi(string(char))
			num = num*10 + atoi
		} else if char == '[' {
			str_stack = append(str_stack, str)
			str = ""
			num_stack = append(num_stack, num)
			num = 0
		} else if char == ']' {
			//	 如果遇见‘]’ 那么str就是 【】之间的数据
			count := num_stack[len(num_stack)-1]
			num_stack = num_stack[:len(num_stack)-1]
			res := str_stack[len(str_stack)-1]
			str_stack = str_stack[:len(str_stack)-1]
			str = res + strings.Repeat(str, count)
		} else {
			// 一直存储的数据
			str += string(char)
		}
	}

	return str
}

// 393. UTF-8 编码验证
func validUtf8(data []int) bool {
	if len(data) == 0 {
		return false
	}
	count := 0
	for i := 0; i < len(data); i++ {
		// 计算字节数
		if count == 0 {
			count = bytelength(data[i])
			// count >0 判断 字节是否合法
		} else {
			// 是否是 10xxxxxx
			if data[i]&(3<<6) != 1<<7 {
				return false
			}
			count--
		}
		// 实际字节数多余count
		if count == -1 {
			return false
		}
	}
	return count == 0

}

func bytelength(data int) int {
	// 0xxxxxxx
	if data>>7 == 0 {
		return 0
	}
	// 110xxxxx 10xxxxxx
	if data>>5 == 6 {

		return 1
	}
	// 1110xxxx 10xxxxxx 10xxxxxx
	if data>>4 == 14 {
		return 2
	}
	// 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
	if data>>3 == 30 {
		return 3
	}
	return -1
}

// 392. 判断子序列
func isSubsequence(s string, t string) bool {

	lenght := len(s)
	if lenght == 0 {
		return true
	}
	start := 0
	for i := 0; i < len(t); i++ {
		if s[start] == t[i] {
			start++
			if start == lenght {
				return true
			}
		}
	}
	return start == lenght

}

// 391. 完美矩形
func isRectangleCover(rectangles [][]int) bool {
	//	 大矩形的面积是左右小矩形的面积之和
	//	所有小矩形组装成大矩形的时候 只有四个顶点不会重合，其他的都会重合，
	//	如果结果不是四个值的话，表示不重合
	type point struct {
		x, y int
	}
	// 最大的矩形
	m := []int{math.MaxInt32, math.MaxInt32, math.MinInt32, math.MinInt32}
	re_map := map[point]int{}
	// 总面积
	sum := 0
	for _, val := range rectangles {
		x, y, a, b := val[0], val[1], val[2], val[3]
		// 四个顶点都存储起来
		re_map[point{x, y}]++
		re_map[point{x, b}]++
		re_map[point{a, b}]++
		re_map[point{a, y}]++
		sum += (val[2] - val[0]) * (val[3] - val[1])
		// 获取矩形最大的四个顶点 计算总面积
		m[0] = min(m[0], x)
		m[1] = min(m[1], y)
		m[2] = max(m[2], a)
		m[3] = max(m[3], b)
	}
	// 总面积不一样 则返回错误
	if sum != (m[2]-m[0])*(m[3]-m[1]) {
		return false
	}
	// 移除四个角 剩下的就是为2的倍数的位置
	re_map[point{m[0], m[1]}]++
	re_map[point{m[0], m[3]}]++
	re_map[point{m[2], m[1]}]++
	re_map[point{m[2], m[3]}]++
	for _, val := range re_map {
		if val != 2 && val != 4 {
			return false
		}
	}
	return true
}

//390. 消除游戏
func lastRemaining(n int) int {
	if n < 3 {
		return n
	}
	if n%2 == 1 {
		return lastRemaining(n - 1)
	} else {
		return 2 * (n/2 + 1 - lastRemaining(n/2))
	}
}

//389. 找不同
func findTheDifference(s string, t string) byte {
	s_map := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		s_map[s[i]]++
	}
	for i := 0; i < len(t); i++ {
		s_map[t[i]]--
		if s_map[t[i]] < 0 {
			return t[i]
		}
	}
	return 0
}

// 387. 字符串中的第一个唯一字符
func firstUniqChar(s string) int {
	m_map := map[int32]int{}
	for _, val := range s {
		m_map[val]++
	}

	for k, val := range s {
		if m_map[val] == 1 {
			return k
		}
	}
	return -1
}

// 384. 打乱数组
type Solution_384 struct {
	// 初始化的数组
	Arr []int
	// 随机打乱的数组
	Cur []int
}

func Constructor_384(nums []int) Solution_384 {
	return Solution_384{Arr: append([]int{}, nums...), Cur: nums}
}

func (this *Solution_384) Reset() []int {
	return this.Arr
}

func (this *Solution_384) Shuffle() []int {
	//	 打乱数组
	for i := 0; i < len(this.Cur)/2; i++ {
		num := rand.Intn(len(this.Cur))
		this.Cur[num], this.Cur[len(this.Cur)-1] = this.Cur[len(this.Cur)-1], this.Cur[num]
	}
	return this.Cur
}

// 383. 赎金信
func canConstruct(ransomNote string, magazine string) bool {
	r_map := map[int32]int{}
	for _, val := range magazine {
		r_map[val]++
	}
	for _, val := range ransomNote {
		r_map[val]--
		if r_map[val] < 0 {
			return false
		}
	}
	return true
}

// 382. 链表随机节点
type Solution_382 struct {
	Arr []int
}

func Constructor_382(head *ListNode) Solution_382 {
	arr := []int{}
	for head != nil {
		arr = append(arr, head.Val)
		head = head.Next
	}
	return Solution_382{
		Arr: arr,
	}
}

func (this *Solution_382) GetRandom() int {
	return this.Arr[rand.Intn(len(this.Arr))]
}

// RandomizedSet 380. O(1) 时间插入、删除和获取随机元素
type RandomizedSet struct {
	//	 key 表示值 value表示在那个下标位置
	r_map map[int]int
	// 存储所有的元素
	r_val []int
}

func Constructor_380() RandomizedSet {
	return RandomizedSet{
		r_map: make(map[int]int),
		r_val: make([]int, 0),
	}

}

func (this *RandomizedSet) Insert(val int) bool {
	_, ok := this.r_map[val]
	if ok {
		return false
	}
	this.r_val = append(this.r_val, val)
	this.r_map[val] = len(this.r_val) - 1
	return true
}

func (this *RandomizedSet) Remove(val int) bool {
	// 获取当前值的位置
	cur, ok := this.r_map[val]
	if !ok {
		return false
	}
	// 将数组原来的最后一位的key的位置 变为需要删除的key对应的val
	this.r_map[this.r_val[len(this.r_val)-1]] = cur
	//	 将数组里面的两个值交换位置
	this.r_val[cur], this.r_val[len(this.r_val)-1] = this.r_val[len(this.r_val)-1], this.r_val[cur]
	this.r_val = this.r_val[:len(this.r_val)-1]
	delete(this.r_map, val)
	return true
}

func (this *RandomizedSet) GetRandom() int {
	return this.r_val[rand.Intn(len(this.r_val))]
}

//378. 有序矩阵中第 K 小的元素
func kthSmallest(matrix [][]int, k int) int {
	arr := []int{}
	for i := 0; i < len(matrix); i++ {
		arr = append(arr, matrix[i]...)
	}
	sort.Ints(arr)
	return arr[k-1]
}

// 377. 组合总和 Ⅳ
func combinationSum4(nums []int, target int) int {
	//	 采用背包问题
	//	k 表示当前的数
	//	当前的数 dp[i] += dp[i-k]
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i <= target; i++ {
		for _, num := range nums {
			// 当i大于num的时候才开始计算
			if i >= num {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}

// 376. 摆动序列
func wiggleMaxLength(nums []int) int {
	//	 采用动态规划
	if len(nums) == 0 {
		return 0
	}
	//	 如果nums[i] >nums[i-1] 那么他的状态就是从上一个 nums[i] < nums[i-1]+1得到当前的数量
	// 如果nums[i]< nums[i-1],当前的数小于上一个数，那么当前的数量就是nums[i]>nums[i-1]+1转移过来的
	//	 准备两个状态，一个是向下的时候的状态，一个是向上的时候的状态
	// 当nums[i]>nums[i-1]
	up := 1
	// 当nums[i] < nums[i-1]
	down := 1
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			up = down + 1
		} else if nums[i] < nums[i-1] {
			down = up + 1
		}
	}
	return max(up, down)
}

//375. 猜数字大小 II
func getMoneyAmount(n int) int {
	// 之所以取n+2 是防止长度溢出
	dp := make([][]int, n+2)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, n+2)
	}
	// 结合遍历开始知道是最小面开始的
	for i := n - 1; i > 0; i-- {
		//	 从对角线右边开始
		for j := i + 1; j <= n; j++ {
			//	 放无穷大 方便取小
			dp[i][j] = math.MaxInt64
			// 计算 i 到j 区间的数据
			for k := i; k <= j; k++ {
				tmp := k + max(dp[i][k-1], dp[k+1][j])
				dp[i][j] = min(dp[i][j], tmp)
			}
		}
	}
	return dp[1][n]
	// dfs
	//memo := make(map[string]int)
	//return getMoneyAmount_dfs(1, n, memo)
}

func getMoneyAmount_dfs(left, right int, memo map[string]int) int {
	key := fmt.Sprintf("%d_%d", left, right)
	if val, ok := memo[key]; ok {
		return val
	}
	if left >= right {
		return 0
	}
	//	 设置一个最大值
	res := math.MaxInt64
	for i := (right + left) / 2; i <= right; i++ {
		// 最小代价的最大值 确保资金够用
		res = min(res, max(getMoneyAmount_dfs(left, i-1, memo), getMoneyAmount_dfs(i+1, right, memo))+i)
	}

	memo[key] = res
	return res
}

// 374. 猜数字大小
func guess_374(num int) int {
	return num
}
func guessNumber(n int) int {
	m := 0
	for m <= n {
		tmpe := m + (n-m)/2
		if guess_374(tmpe) == 0 {
			return tmpe
		} else if guess_374(tmpe) < 0 {
			n = tmpe - 1
		} else {
			m = tmpe + 1
		}
	}
	return m

}

// 373. 查找和最小的K对数字
func kSmallestPairs(nums1 []int, nums2 []int, k int) [][]int {
	heap := NewHeap()
	for _, val := range nums1 {
		for _, value := range nums2 {
			heap.PushMax([]int{val, value})
			if heap.Count() == k {
				return heap.data[1:]
			}
		}
	}
	list := heap.data[1:]
	return list
}

type Heap_373 struct {
	data  [][]int
	count int
}

// NewHeap 初始化堆，第一个元素不用
func NewHeap() *Heap_373 {
	// 第一个位置空着，不使用
	arr := [][]int{{0, 0}}
	return &Heap_373{data: arr, count: 0}
}

// Parent 父节点的位置
func (h *Heap_373) Parent(root int) int {
	return root / 2
}

// 左子树的位置
func (h *Heap_373) left(root int) int {
	return root * 2
}

// 右子树的位置
func (h *Heap_373) right(root int) int {
	return root*2 + 1
}

// Pop 返回堆顶元素
func (h *Heap_373) Pop() []int {
	return h.data[1]
}

// 堆化的时候交换元素
func (h *Heap_373) exchange(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *Heap_373) Count() int {
	return h.count
}

// 大根堆操作

// PushMax 新增元素 大根堆
func (h *Heap_373) PushMax(v []int) {
	h.count++
	h.data = append(h.data, v)
	// 元素新增之后进行堆化
	h.swimMax(h.count)
}

// PushListMax 在大根堆里面新增多个元素
func (h *Heap_373) PushListMax(arr [][]int) {
	for _, value := range arr {
		h.PushMax(value)
	}
}

// DelMax 删除大根堆的头部元素
func (h *Heap_373) DelMax() []int {
	max := h.data[1]
	// 将最后一个元素放到第一个堆顶位置，然后向下进行堆化
	h.exchange(1, h.count)
	h.data = h.data[:h.count]
	h.count--
	h.sinkMax(1)
	return max
}

// 向上堆化 大根堆
func (h *Heap_373) swimMax(key int) {
	// 根节点<当前节点
	// 堆化
	for key > 1 && h.data[key][0]+h.data[key][1] > h.data[h.Parent(key)][0]+h.data[h.Parent(key)][1] {
		h.exchange(h.Parent(key), key)
		key = h.Parent(key)
	}
}

//向下进行堆化
func (h *Heap_373) sinkMax(key int) {
	// 下沉到堆底
	for h.left(key) <= h.count {
		order := h.left(key)
		if h.right(key) <= h.count && h.data[order][0]+h.data[order][1] < h.data[h.right(key)][0]+h.data[h.right(key)][1] {
			order = h.right(key)
		}
		// 节点比两个子节点都大,就不必下沉了
		if h.data[order][0]+h.data[order][1] < h.data[key][0]+h.data[key][1] {
			break
		}
		h.exchange(key, order)
		key = order
	}
}
