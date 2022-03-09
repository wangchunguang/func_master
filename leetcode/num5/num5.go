package num5

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
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
	return &TreeNode{1,
		&TreeNode{2, nil, &TreeNode{5, nil, nil}},
		&TreeNode{3, nil, &TreeNode{4, nil, nil}}}

}

func AddNode() *Node {

	node := &Node{1, &Node{2, &Node{6, &Node{3, &Node{4, nil, nil}, nil}, nil}, nil}, nil}

	return node
}

func main() {
	// 下一次出现排序 采用堆排序
	containsNearbyDuplicate([]int{1, 2, 3, 1, 2, 3}, 2)
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

// 217. 存在重复元素
func containsDuplicate(nums []int) bool {
	sort.Ints(nums)
	for i := 0; i < len(nums)-1; i++ {
		if nums[i] == nums[i+1] {
			return true
		}

	}
	return false
}

// 216  组合总和 III
func combinationSum3(k int, n int) [][]int {
	//	 采用回溯算法
	arr := [][]int{}
	combinationSum3_dfs(k, n, 1, []int{}, &arr)
	fmt.Println(arr)
	return arr
}

func combinationSum3_dfs(k, n, start int, path []int, arr *[][]int) {
	sum := 0
	for _, value := range path {
		sum += value
	}
	if sum > n {
		return
	}
	if sum == n && k == len(path) {
		cur := make([]int, len(path))
		copy(cur, path)
		*arr = append(*arr, cur)
		return
	}

	for i := start; i < 10; i++ {
		//	 剪枝
		if sum+i > n {
			continue
		}
		// 回溯
		path = append(path, i)
		combinationSum3_dfs(k, n, i+1, path, arr)
		path = path[:len(path)-1]
	}
}

// 215 数组中的第K个最大元素
func findKthLargest(nums []int, k int) int {
	// 采用快速排序实现
	QuickSort(nums, 0, len(nums)-1)
	return nums[len(nums)-k+1]
}

func QuickSort(nums []int, left, right int) {
	if left < right {
		// 每一次的partition方法 就是将数组划分为两部分 大于prive  还有小于 prive的左右两部分
		// 将左右的两部分进行循环
		prive := partition(nums, left, right)
		// 递归计算
		QuickSort(nums, left, prive-1)
		QuickSort(nums, prive+1, right)
	}
}

func partition(nums []int, left, right int) int {
	prive := nums[left]
	for left < right {
		// 开始循环 判断数组右边的数依次向左移动 找到小于指定的中间数
		for left < right && prive <= nums[right] {
			right--
		}
		// 因为nums[left]已经提取出来 所以将left的数据填入小于prive的数
		nums[left] = nums[right]
		for left < right && prive >= nums[left] {
			left++
		}
		nums[right] = nums[left]
	}
	nums[left] = prive
	return left
}

// 214 最短回文串
// 暴力解法是将原来的字符串倒叙 然后一个一个拿出来加入原来的字符串，如果符合要求 就是最少的回文字符串
func shortestPalindrome(s string) string {
	if len(s) == 1 {
		return s
	}
	// 判断是否是回文
	flag := true
	// 先判断本身是不是回文
	start, end := 0, len(s)-1
	for start < end {
		if s[start] != s[end] {
			flag = false
			break
		}
		start++
		end--
		flag = true
	}
	if flag {
		return s
	}
	var buffer bytes.Buffer
	for i := len(s) - 1; i >= 0; i-- {
		buffer.WriteString(string(s[i]))
	}
	// 将字符串逆序
	str := buffer.String()

	s3 := ""
	// 开始遍历数据
	for i := 1; i <= len(str); i++ {
		s2 := str[:i] + s
		//	 判断s2是不是回文 是的话 直接返回
		l, r := 0, len(s2)-1
		for l < r {
			if s2[l] != s2[r] {
				flag = false
				break
			}
			l++
			r--
			flag = true
		}
		//	 如果
		if flag {
			s3 = s2
			break
		}
	}
	return s3
}

//  213  打家劫舍 II
func rob2(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	// 表示偷盗第一间房 那么就不会偷最后一间房子
	dp1 := make([]int, len(nums))
	// 表示不偷盗第一间房子 那么就可以偷盗最后一间房子
	dp2 := make([]int, len(nums))
	dp1[0] = nums[0]
	dp1[1] = max(nums[0], nums[1])
	for i := 2; i < len(nums)-1; i++ {
		dp1[i] = max(dp1[i-2]+nums[i], dp1[i-1])
	}
	dp2[0] = 0
	dp2[1] = nums[1]
	for i := 2; i < len(nums); i++ {
		dp2[i] = max(dp2[i-2]+nums[i], dp2[i-1])
	}

	return max(dp2[len(nums)-1], dp1[len(nums)-2])
}

//212 单词搜索采用回溯算法实现
func findWords(board [][]byte, words []string) []string {
	arr := []string{}
	trie := Constructor_Trie()
	// 将需要查找的words数组里面的单词全部添加到前缀树
	for _, value := range words {
		trie.Insert(value)
	}
	// 因为每个字符串不能重复的使用单元格
	flag := make([][]bool, len(board))
	for i := 0; i < len(board); i++ {
		flag[i] = make([]bool, len(board[0]))
	}
	path := []byte{}
	// 判断每一个字符是否符合要求
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			findWords_dfs(i, j, board, &arr, &path, &trie, flag)
		}
	}
	return arr
}

// i j 表示二维数组的坐标 board 表示二维数组 arr表示需要添加进去的字符  path表示出现的字符
func findWords_dfs(i, j int, board [][]byte, arr *[]string, path *[]byte, trie *Trie, flag [][]bool) {
	//	 边界条件
	if i < 0 || j < 0 || i > len(board)-1 || j > len(board[0])-1 || flag[i][j] == true {
		return
	}
	// 获取当前的字符
	ch := board[i][j]
	// 当数据不存在的时候
	if trie.ListTrie[ch-'a'] == nil {
		return
	}
	//	 当前数据存在的时候 将字符变为下次不可使用
	flag[i][j] = true
	// 将数据添加到组合里面
	*path = append(*path, ch)
	trie = trie.ListTrie[ch-'a']
	// 判断是否是当前位置结尾
	if trie.IsTrie {

		*arr = append(*arr, string(*path))
		trie.IsTrie = false
	}
	// 判断走那些路径 前后左右
	directions := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for _, value := range directions {
		findWords_dfs(i+value[0], j+value[1], board, arr, path, trie, flag)
	}

	*path = (*path)[:len(*path)-1]
	// 回溯
	flag[i][j] = false
}

// Trie 前缀树
type Trie struct {
	// 是否是某个单词的结尾
	IsTrie bool
	// 表示当前字符下一个字符有哪些
	ListTrie [26]*Trie
}

func Constructor_Trie() Trie {
	return Trie{}
}

func (this *Trie) Insert(word string) {

	for _, value := range word {
		if this.ListTrie[value-'a'] == nil { //判断第一层是否有这个数据
			this.ListTrie[value-'a'] = &Trie{}
		}
		// 向下面一层查找
		this = this.ListTrie[value-'a']
	}
	// 当一个单词结束的时候指定这个的结尾
	this.IsTrie = true
}

func (this *Trie) Search(word string) bool {
	for _, value := range word {
		// 当前层判断是否有这个数据 如果没有直接返回
		if this.ListTrie[value-'a'] == nil {
			return false
		}
		// 进行下一层寻找
		this = this.ListTrie[value-'a']
	}
	// 判断是不是结尾
	return this.IsTrie
}

func (this *Trie) StartsWith(prefix string) bool {
	for _, value := range prefix {
		if this.ListTrie[value-'a'] == nil {
			return false
		}
		this = this.ListTrie[value-'a']
	}
	return true
}

// 210. 课程表 II
func findOrder(numCourses int, prerequisites [][]int) []int {
	//创建适合图的数据结构 一位数组表示当前的数据 二维数组里面的数据表示出度所到达的位置
	t := make([][]int, numCourses)
	// 当前的数据有多少的入度
	t_entry := make([]int, numCourses)
	// 学习结束的课程
	res := make([]int, 0)
	for _, value := range prerequisites {
		// 采用领接表的形式
		t[value[1]] = append(t[value[1]], value[0])
		// 入度++
		t_entry[value[0]]++
	}
	// 创建一个队列形式,将所有入度为0的数据添加到队列中
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if t_entry[i] == 0 {
			queue = append(queue, i)
		}

	}
	for len(queue) > 0 {
		node := queue[0]
		res = append(res, node)
		queue = queue[1:]
		//	 获取当前数据出度的数据 例如 1-》2-》3
		for _, value := range t[node] {
			t_entry[value]--
			if t_entry[value] == 0 {
				queue = append(queue, value)
			}
		}
	}
	// 判断是否有环
	if len(res) != numCourses {
		return []int{}
	}
	return res
}

// 209. 长度最小的子数组
func minSubArrayLen(target int, nums []int) int {
	lenght := len(nums)
	if lenght == 0 {
		return 0
	}
	left, right := 0, 0
	// 当前子数组的累加和
	sum := nums[0]
	// 当前子数组最小的长度
	arr_size := lenght + 1
	for left <= right && right < lenght {
		// 如果开始的值大于target 那么left ++
		if sum >= target {
			size := len(nums[left : right+1])
			if size == 1 {
				return 1
			}
			// 获取字符串最小的长度
			arr_size = min(arr_size, size)
			sum -= nums[left]
			left++
		} else { // 如果小于的话 就进行加加
			right++
			if right < lenght {
				sum += nums[right]
			}

		}
	}
	if arr_size != lenght+1 {
		return arr_size
	}
	return 0
}

// 207 课程表
func canFinish(numCourses int, prerequisites [][]int) bool {
	//	表示存储节点的出度，多少行，表示是哪个课程，有多少列 表示就有多少课程是它的出度
	node := make([][]int, numCourses)
	// 入度的数量
	entry_num := make([]int, numCourses)
	//	表示有多少课程学习完了
	res := make([]int, 0, numCourses)
	// 将课程初始化 判断每个课程的出度有哪些 入度的数量
	for _, value := range prerequisites {
		// 因为要学习value【0】 就要先完成value【1】的课程
		// 所以value【0】的入度就是value怕【1】
		// value【1】的出度就是value【0】
		node[value[1]] = append(node[value[1]], value[0])
		entry_num[value[0]]++
	}
	//	 创建一个队列 将所有入度为0的数据 存入到队列中
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		// 判断这门课的入度是不是为0
		if entry_num[i] == 0 {
			queue = append(queue, i)
		}
	}

	for len(queue) > 0 {
		//	 获取队列的头部
		start := queue[0]
		res = append(res, start)
		queue = queue[1:]
		//	 获取当前队列头部，将数据存入队列 并且将它的所有指向的数据 将那些数据-1
		for _, value := range node[start] {
			entry_num[value]--
			if entry_num[value] == 0 {
				queue = append(queue, value)
			}
		}
	}
	return len(res) == numCourses
}

// 反转链表
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	var root *ListNode
	cur := head
	for cur != nil {
		next := cur.Next
		cur.Next = root
		root = cur
		cur = next
	}

	return root

}

func countPrimes(n int) int {
	count := 0
	signs := make([]bool, n)
	for i := 2; i < n; i++ {
		if signs[i] {
			continue
		}
		count++
		// 排除是是偶数的情况
		// 当前的数乘以2   并且依次加上初始的数值1
		for j := 2 * i; j < n; j += i {
			signs[j] = true
		}
	}
	return count
}

// 删除链表元素
func removeElements(head *ListNode, val int) *ListNode {
	if head == nil {
		return head
	}
	next := &ListNode{Val: 0}
	next.Next = head
	prev := next
	for prev.Next != nil {
		if prev.Next.Val == val {
			prev.Next = prev.Next.Next
		} else {
			prev = prev.Next
		}
	}
	return next.Next
}

func step(n int) int {
	sum := 0
	for n > 0 {
		// 计算个位数
		sum += (n % 10) * (n % 10)
		n = n / 10
	}
	return sum
}

// 202. 快乐数
func isHappy(n int) bool {
	m := make(map[int]bool)
	// 等于1 为快乐数 当不为1 并且没有出现过这个数就可以进行下一次轮询
	for n != 1 && !m[n] {
		n = step(n)
		m[n] = true
	}
	return n == 1
}

//  200. 岛屿数量 dfs
func numIslands(grid [][]byte) int {
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				numIslands_dfs(grid, i, j)
				count++
			}

		}
	}
	return count
}

func numIslands_dfs(grid [][]byte, i, j int) {
	// 不符合要求
	if i < 0 || j < 0 || i > len(grid)-1 || j > len(grid[0])-1 {
		return
	}
	//	 如果不是岛屿的话 返回
	if grid[i][j] == '1' {
		return
	}
	//	 将遇见的岛屿修改成2
	grid[i][j] = '2'
	numIslands_dfs(grid, i+1, j)
	numIslands_dfs(grid, i-1, j)
	numIslands_dfs(grid, i, j+1)
	numIslands_dfs(grid, i, j-1)
}

//199. 二叉树的右视图
func rightSideView(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	// 宽度优先遍历 选最右边的一个
	arr := []int{root.Val}
	list := []*TreeNode{root}
	//	表示当前节点最后的点
	var nodeCurend *TreeNode
	// 表示当前层数据的位置
	var nodenextend *TreeNode
	nodeCurend = root
	for len(list) != 0 {
		list_root := list[0]
		list = list[1:]
		if list_root.Right != nil {
			list = append(list, list_root.Right)
			nodenextend = list_root.Right
		}
		if list_root.Left != nil {
			list = append(list, list_root.Left)
			nodenextend = list_root.Left
		}
		if nodeCurend == list_root {
			nodeCurend = nodenextend
			nodenextend = nil
			if len(list) > 0 {
				arr = append(arr, list[0].Val)
			}
		}
	}
	return arr
}

// 198. 打家劫舍
func rob(nums []int) int {
	if len(nums) < 2 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	dp[1] = nums[1]
	if len(nums) == 2 {
		if dp[0] > dp[1] {
			return dp[0]
		} else {
			return dp[1]
		}
	}
	max_num := nums[0]
	for i := 2; i < len(nums); i++ {
		dp[i] = max(nums[i]+max_num, dp[i-1])
		max_num = max(max_num, dp[i-1])
	}
	return dp[len(nums)-1]
}

// 189. 旋转数组
func rotate(nums []int, k int) {
	n := len(nums)
	k = k % n
	run(nums[:n-k])
	run(nums[n-k:])
	run(nums)
}

func run(nums []int) {
	l, r := 0, len(nums)-1
	for l < r {
		nums[l], nums[r] = nums[r], nums[l]
		l++
		r--
	}
}

// 187. 重复的DNA序列
func findRepeatedDnaSequences(s string) (res []string) {
	m := map[string]int{}
	l := len(s)
	for i := 0; i <= l-10; i++ {
		m[s[i:i+10]]++
	}

	for k, v := range m {
		if v > 1 {
			res = append(res, k)
		}
	}

	return
}

// KMP 处理一个kmp算法
// s1 表示需要匹配的字符串 s2 表示模式串
func KMP(haystack, needle string) bool {
	prefix := KMP_prefix(needle)
	l1, l2 := len(haystack), len(needle)
	i, j := 0, 0
	for i < l1 && j < l2 {
		if haystack[i] == needle[j] {
			i++
			j++
		} else if j == 0 { // 如果等于0的位置 表示没办法继续比了 将主串++
			i++
		} else {
			// 如果两个值不等 那么就查找是否还有前面的子串
			j = prefix[j]
		}
	}
	if l2 == j {
		return true
	}

	return false
}

// KMP_prefix 获取模式串的最长前缀
func KMP_prefix(str string) []int {
	lenght := len(str)
	if lenght == 1 {
		return []int{-1}
	}
	// 创建一样长的数组
	arr := make([]int, lenght)
	// 第一位置和第二位的可以默认设置
	arr[0] = -1
	arr[1] = 0
	// 因为最长前缀是不包含它自己本身的情况
	// 比如 abcde，e的前缀就是在abcd中寻找
	// 设置移动的值和初始值
	// i表示前缀的地址 2表示字符开始的位置
	i, j := 0, 2
	for j < lenght {
		// 当前值的前一个值与前缀开始的地址去对比
		if str[j-1] == str[i] {
			//  前缀和后缀一样，向后移动
			i++
			// 赋值
			arr[j] = i
			// 继续对比下一个字符
			j++
		} else if i > 0 {
			// 如果当前值的前一个与它最大前缀的下一个值不等
			// 那么就判断它前缀的前缀的值
			i = arr[i]
		} else {
			// 如果等于0  就直接设置为0
			arr[j] = 0
			j++
		}
	}

	return arr
}

// 179. 最大数
func largestNumber(nums []int) string {
	// 将数据转为字符串 然后将字符串排序
	str_arr := make([]string, len(nums))
	for i := 0; i < len(nums); i++ {
		str_arr[i] = strconv.Itoa(nums[i])
	}
	sort.Slice(str_arr, func(i, j int) bool {
		return str_arr[i]+str_arr[j] >= str_arr[j]+str_arr[i]
	})
	//sort.Sort(sort.Reverse(sort.StringSlice(str_arr)))
	o := strings.Join(str_arr, "")
	if o[0] == '0' {
		return "0"
	}
	return o
}

func calculateMinimumHP_174(dungeon [][]int) int {
	//将行和列的长度
	row, list := len(dungeon), len(dungeon[0])
	// 初始化二维数组
	dp := make([][]int, row)
	for i := 0; i < row; i++ {
		dp[i] = make([]int, list)
	}
	// 因为从第一位开始向下没办法做动态规划，所以尝试从最后一位倒着推
	// 如果公主在的位置为整数，表示最后一步不需要健康点数，如果为负数就表示需要健康点数
	// 因为健康点数必须大于1 所以取相反数，取大值
	dp[row-1][list-1] = max(1, 1-dungeon[row-1][list-1])

	// 初始化最后一行的数据 因为最后一行的最后一列的数据已经有数据，所以从倒数第二行进行计算
	for i := list - 2; i >= 0; i-- {
		// 如果计算出来为负数，表示左边健康点数为正整数，因为采用的是最少的健康点数，所以为1，
		// 如果计算出来的为正整数，表示左边的数原来就是负数，所以需要的总的健康点数，就是两个值相加，负负得正
		dp[row-1][i] = max(1, dp[row-1][i+1]-dungeon[row-1][i])
	}
	// 初始化最后一列的数据
	for i := row - 2; i >= 0; i-- {
		dp[i][list-1] = max(1, dp[i+1][list-1]-dungeon[i][list-1])
	}
	for i := row - 2; i >= 0; i-- {
		for j := list - 2; j >= 0; j-- {
			// 选择左边和右边的较小值
			tmp := min(dp[i][j+1], dp[i+1][j])
			dp[i][j] = max(1, tmp-dungeon[i][j])
		}
	}
	return dp[0][0]
}

// 174. 地下城游戏
func calculateMinimumHP(dungeon [][]int) int {
	row, list := len(dungeon), len(dungeon[0])
	// 先将最上面的计算出来
	dp := make([][]int, row)
	for i := 0; i < row; i++ {
		dp[i] = make([]int, list)
	}
	// 如果最后一个为负数则表示需要多少健康点 如果为正整数表示只需要1
	// 因为最开始的初始就需要一步
	dp[row-1][list-1] = max(1, 1-dungeon[row-1][list-1])
	// 初始化最后一行 如果为负数，表示右边的位置比当前位置小，所以可以直接设置为1
	for i := list - 2; i >= 0; i-- {
		dp[row-1][i] = max(1, dp[row-1][i+1]-dungeon[row-1][i])
	}
	// 最后一列 如果为负数表示当前值比下面的值大， 那么就设置为1
	for i := row - 2; i >= 0; i-- {
		dp[i][list-1] = max(1, dp[i+1][list-1]-dungeon[i][list-1])
	}
	for i := row - 2; i >= 0; i-- {
		for j := list - 2; j >= 0; j-- {
			//	 判断下方 和右边 谁小选择谁
			tmp := min(dp[i+1][j], dp[i][j+1])
			// 如果当前位置的值小于 右边 下边最小的值 则设置为-1
			dp[i][j] = max(1, tmp-dungeon[i][j])
		}
	}

	return dp[0][0]
}

type BSTIterator struct {
	Arr []int
}

func Constructor_173(root *TreeNode) BSTIterator {
	arr := []int{}
	Constructor_BST(root, &arr)
	return BSTIterator{Arr: arr}
}

func Constructor_BST(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	Constructor_BST(root.Left, arr)
	*arr = append(*arr, root.Val)
	Constructor_BST(root.Right, arr)
}

func (this *BSTIterator) Next() int {
	if len(this.Arr) == 0 {
		return 0
	}
	num := this.Arr[0]
	this.Arr = this.Arr[1:]
	return num
}

func (this *BSTIterator) HasNext() bool {
	if len(this.Arr) > 0 {
		return true
	}
	return false
}

//172. 阶乘后的零
func trailingZeroes(n int) int {
	ans := 0
	for n >= 5 {
		ans = ans + n/5
		n = n / 5
	}
	return ans
}

//  171. Excel 表列序号
func titleToNumber(columnTitle string) int {

	ans := 0
	for i := 0; i < len(columnTitle); i++ {
		ans = ans*26 + int((columnTitle[i])-64)
	}
	return ans
}

// 85. 最大矩形
func maximalRectangle(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}
	//	 将每一行有多少1 先求出来
	dp := make([][]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[0]))
	}

	for i := 0; i < len(matrix); i++ {
		for j := len(matrix[0]) - 1; j >= 0; j-- {
			//	 判断是不是最后的一个
			if j == len(matrix[0])-1 {
				if matrix[i][j] == '1' {
					dp[i][j] = 1
				} else {
					dp[i][j] = 0
				}
			} else if matrix[i][j] == '1' {
				dp[i][j] = dp[i][j+1] + 1
			} else {
				dp[i][j] = 0
			}
		}
	}
	num_max := 0
	for i := 0; i < len(matrix); i++ {
		// 一层一层的进行判断
		for j := len(matrix[0]) - 1; j >= 0; j-- {
			min_num := dp[i][j]
			if min_num == 0 {
				continue
			}
			//	 每一个就是向下递推，选最小的一个长度 并且乘上这个高度
			for k := i; k < len(matrix); k++ {
				if dp[k][j] == 0 {
					break
				}
				min_num = min(min_num, dp[k][j])
				num_max = max((k-i+1)*min_num, num_max)
			}
		}
	}
	return num_max
}

// 169 摩尔算法
func majorityElement_169(nums []int) int {
	//	 摩尔算法只适合用在票数过半的情况下
	//	设置count 当遇见的一样的 ++ 不一样的--
	//	因为要求是票数过半 所以最后剩下的那个就是满足条件的数
	num, count := 0, 0

	for _, value := range nums {
		if count == 0 {
			num = value
		}
		if value == num {
			count++
		} else {
			count--
		}
	}
	return num
}

//  169. 多数元素
func majorityElement(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	n_map := make(map[int]int)
	age := len(nums) / 2
	for _, value := range nums {
		if val, ok := n_map[value]; ok {
			val += 1
			if age < val {
				return value
			}
			n_map[value] = val
		} else {
			n_map[value] = 1
		}
	}
	return 0
}

// 168. Excel表列名称
func convertToTitle(columnNumber int) string {
	str := ""
	for columnNumber != 0 {
		num := columnNumber % 26
		if num == 0 {
			num += 26
			columnNumber -= 26
		}
		str = string(byte(num)+64) + str
		columnNumber = columnNumber / 26
	}
	fmt.Println(str)
	return str
}

// 采用双指针方式计算
func twoSum_167(numbers []int, target int) []int {
	arr := []int{}
	l, r := 0, len(numbers)-1
	for l < r {
		if (numbers[l] + numbers[r]) > target {
			r--
		} else if (numbers[l] + numbers[r]) < target {
			l++
		} else {
			arr = append(arr, l+1)
			arr = append(arr, r+1)
			return arr
		}
	}
	return arr
}

// 167. 两数之和 II - 输入有序数组 采用哈希表方式计算
func twoSum(numbers []int, target int) []int {
	n_map := make(map[int]int)
	for i := 0; i < len(numbers); i++ {
		n_map[numbers[i]] = i
	}
	arr := []int{}
	for i := 0; i < len(numbers); i++ {
		num := target - numbers[i]
		if val, ok := n_map[num]; ok {
			arr = append(arr, i+1)
			arr = append(arr, val+1)
			return arr
		}
		if numbers[i] > target {
			return arr
		}
	}
	return arr
}

// 166 分数到小数
func fractionToDecimal(numerator int, denominator int) string {
	// 判断边界条件
	if numerator == 0 {
		return "0"
	}
	if denominator == 0 {
		return ""
	}
	var buffer bytes.Buffer
	//	 如果其中的一个为负数 那么结果就为负数
	if (numerator > 0 && denominator < 0) || (numerator < 0 && denominator > 0) {
		buffer.WriteString("-")
	}
	num := int(math.Abs(float64(numerator)))
	denom := int(math.Abs(float64(denominator)))
	buffer.WriteString(strconv.Itoa(num / denom))
	num = num % denom
	if num == 0 {
		return buffer.String()
	}
	//	 小数点前面的数在上面已经算出来了
	buffer.WriteString(".")
	//	 通过hash表存储已经出现过的小数
	num_m := make(map[int]int)

	repeatPos := -1
	for {
		num = num * 10
		pos, ok := num_m[num]
		if !ok {
			num_m[num] = buffer.Len()
		} else {
			repeatPos = pos
			break
		}
		buffer.WriteString(strconv.Itoa(num / denom))
		num = num % denom
		if num == 0 {
			break
		}
	}
	if repeatPos == -1 {
		return buffer.String()
	}
	res := buffer.String()
	fmt.Println(res)
	return fmt.Sprintf("%s(%s)", res[0:repeatPos], res[repeatPos:])
}

//165. 比较版本号
func compareVersion(version1 string, version2 string) int {
	//	 比较版本号 就是将两个通过。转换为数组
	arr1 := strings.Split(version1, ".")
	arr2 := strings.Split(version2, ".")
	//	将短的数组后续补0
	if len(arr1) < len(arr2) {
		size := len(arr2) - len(arr1)
		for i := 0; i < size; i++ {
			arr1 = append(arr1, "0")
		}
	} else {
		size := len(arr1) - len(arr2)
		for i := 0; i < size; i++ {
			arr2 = append(arr2, "0")
		}

	}
	//	 将两个数组成为一样长
	//	然后遍历匹配就可以了
	for i := 0; i < len(arr2); i++ {
		num1, _ := strconv.Atoi(arr1[i])
		num2, _ := strconv.Atoi(arr2[i])
		if num1 < num2 {
			return -1
		} else if num1 > num2 {
			return 1
		}
	}
	return 0
}

// 164 最大间距
func maximumGap(nums []int) int {
	if len(nums) < 2 {
		return 0
	}
	arr := mergeSort_164(nums)
	max_num := 0
	for i := 1; i < len(arr); i++ {
		max_num = max(max_num, arr[i]-arr[i-1])
	}
	return max_num
}

// 尝试归并排序
func mergeSort_164(nums []int) []int {
	//	 将数据拆分 因为归并是由上到下的，所以进行分组
	if len(nums) <= 1 {
		return nums
	}
	mid := len(nums) / 2
	left := mergeSort_164(nums[:mid])
	right := mergeSort_164(nums[mid:])
	return merge_164(left, right)
}

// 数据进行拆分只有排序
func merge_164(left, right []int) []int {
	l, r := 0, 0
	arr := []int{}
	for l < len(left) && r < len(right) {
		if left[l] < right[r] {
			arr = append(arr, left[l])
			l++
		} else {
			arr = append(arr, right[r])
			r++
		}
	}
	if l < len(left) {
		arr = append(arr, left[l:]...)
	}
	if r < len(right) {
		arr = append(arr, right[r:]...)
	}
	return arr
}

// 162 寻找峰值 二分查找法
func findPeakElement_162(nums []int) int {
	l, r := 0, len(nums)-1
	for l < r {
		mid := l + (r-l)/2
		// 右边有峰值
		if nums[mid] < nums[mid+1] {
			l = mid + 1
		} else {
			r = mid
		}
	}
	return r
}

// 162 寻找峰值
func findPeakElement(nums []int) int {
	if len(nums) == 0 || len(nums) == 1 {
		return 0
	}
	if len(nums) == 2 {
		if nums[0] > nums[1] {
			return 0
		} else {
			return 1
		}
	}
	for i := 1; i < len(nums)-1; i++ {
		if nums[i] > nums[i-1] && nums[i] > nums[i+1] {
			return i
		}
	}
	if nums[len(nums)-1] > nums[len(nums)-2] {
		return len(nums) - 1
	}
	return 0
}

// 160 判断链表是否相交
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	//	 判断两个链表的长度

	A := headA
	B := headB
	for A != B {
		if A == nil {
			A = headB
		} else {
			A = A.Next
		}
		if B == nil {
			B = headA
		} else {
			B = B.Next
		}

	}
	return A
}

type MinStack struct {
	Value []int
}

func Constructor_160() MinStack {
	return MinStack{Value: []int{}}
}

func (this *MinStack) Push(val int) {
	this.Value = append(this.Value, val)

}

func (this *MinStack) Pop() {
	this.Value = this.Value[:len(this.Value)-1]
}

func (this *MinStack) Top() int {
	num := this.Value[len(this.Value)-1]
	return num

}

func (this *MinStack) GetMin() int {
	if len(this.Value) == 0 {
		return -1
	}
	min_num := math.MaxInt64
	for i := 0; i < len(this.Value); i++ {
		min_num = min(this.Value[i], min_num)
	}
	return min_num

}

// 螺旋数组解法
func spiralOrder(matrix [][]int) []int {
	//	 确定左上角和右下角的坐标
	if len(matrix) == 0 {
		return nil
	}

	ax, ay, bx, by := 0, 0, len(matrix)-1, len(matrix[0])-1
	arr := []int{}

	// 左上角向右下角移动 右下角向左上角移动
	for ax <= bx && ay <= by {
		spiralOrder_54(ax, ay, bx, by, &arr, matrix)
		ax, ay, bx, by = ax+1, ay+1, bx-1, by-1
	}
	return arr
}

func spiralOrder_54(ax, ay, bx, by int, arr *[]int, matrix [][]int) {
	//	 当两个点的x轴相等的时候 表示最后一行了
	if ax == bx {
		for i := ay; i <= by; i++ {
			*arr = append(*arr, matrix[ax][i])
		}
	} else if ay == by { // 如果只有一列了
		for i := ax; i <= bx; i++ {
			*arr = append(*arr, matrix[i][ax])
		}
	} else {
		for i := ay; i < by; i++ {
			*arr = append(*arr, matrix[ax][i])
		}
		for i := ax; i < bx; i++ {
			*arr = append(*arr, matrix[i][by])
		}
		for i := by; i > ay; i-- {
			*arr = append(*arr, matrix[bx][i])
		}
		for i := bx; i > ax; i-- {
			*arr = append(*arr, matrix[i][ay])
		}
	}
}

// 154. 寻找旋转排序数组中的最小值
func findMin_154(nums []int) int {
	min_num := math.MaxInt64
	for i := 0; i < len(nums); i++ {
		min_num = min(min_num, nums[i])

	}
	return min_num
}

// 153. 寻找旋转排序数组中的最小值
func findMin(nums []int) int {
	min_num := math.MaxInt64
	for i := 0; i < len(nums); i++ {
		min_num = min(min_num, nums[i])

	}
	return min_num
}

// 152 乘机最大的子数组
func maxProduct(nums []int) int {
	//	 设置两个数 可能是最大的 也可能是最小的数
	max_num, min_num := 1, 1
	res := nums[0]
	for i := 0; i < len(nums); i++ {
		// 可能当前的数是整数
		max_num = max(max(max_num*nums[i], min_num*nums[i]), nums[i])
		// 可能当前的是负数
		min_num = min(min(max_num*nums[i], min_num*nums[i]), nums[i])
		res = max(max_num, res)
	}
	return res
}

//151. 翻转字符串里的单词
func reverseWords(s string) string {
	lenght := len(s)
	arr := []string{}
	str := ""
	for i := 0; i < lenght; i++ {
		//	 判断是空格的时候将数据添加进
		if s[i] != 32 {
			str += string(s[i])
		} else if len(str) > 0 {
			arr = append([]string{str}, arr...)
			str = ""
		}
	}
	if len(str) > 0 {
		arr = append([]string{str}, arr...)
	}
	s1 := ""
	for _, value := range arr {
		s1 += value + " "
	}
	return s1[:len(s1)-1]
}

// 150. 逆波兰表达式求值
func evalRPN(tokens []string) int {
	// 将集中运算符标记出来
	t_map := make(map[string]string)
	t_map["+"] = "+"
	t_map["-"] = "-"
	t_map["*"] = "*"
	t_map["/"] = "/"
	statck := []int{}
	// 采用栈的方式进行计算
	for _, value := range tokens {
		if _, ok := t_map[value]; ok {
			num1 := statck[len(statck)-1]
			num2 := statck[len(statck)-2]
			statck = statck[:len(statck)-2]
			switch value {
			case "*":
				statck = append(statck, num2*num1)
			case "-":
				statck = append(statck, num2-num1)
			case "+":
				statck = append(statck, num2+num1)
			case "/":
				statck = append(statck, num2/num1)
			}
		} else if num, err := strconv.Atoi(value); err == nil {
			// 如果是纯数字 将数字进栈
			statck = append(statck, num)
		}
	}
	return statck[0]

}

// 直线上最多的点数 149
func maxPoints(points [][]int) int {
	//	 进行三次循环 先选取两个点 然后再选取第三个点判断是否符合条件
	lengh := len(points)
	ans := 1
	for i := 0; i < lengh; i++ {
		x := points[i]
		for j := i + 1; j < lengh; j++ {
			y := points[j]
			count := 2
			for k := j + 1; k < lengh; k++ {
				z := points[k]
				// 假设点为 x(x0,x1) y(y0,y1) z(z0,z1)
				// 斜率公式 （x1-y1)/(x0-y0) = (x1-z1)/(x0-z0)
				// 因为0没法为除数，所以转换成乘法的公式
				// （x1-y1) * (x0-z0) = (x1-z1) * (x0-y0)
				// 计算斜率
				s1 := (x[1] - z[1]) * (x[0] - y[0])
				s2 := (x[1] - y[1]) * (x[0] - z[0])
				//s1 := (y[1] - x[1]) * (p[0] - y[0])
				//s2 := (p[1] - y[1]) * (y[0] - x[0])
				if s1 == s2 {
					count++
				}
			}
			ans = max(ans, count)
		}
	}
	return ans
}

// 148 链表的并归排序
func sortList_148(head *ListNode) *ListNode {
	// 递归的出口
	if head == nil || head.Next == nil {
		return head
	}
	// 快慢指针
	slow, fast := head, head
	// 保存慢指针的前一个节点
	var pre *ListNode

	for fast != nil && fast.Next != nil {
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	// 将数据分成两段
	pre.Next = nil
	list := sortList_148(head)
	right := sortList_148(slow)

	return mergeList_148(list, right)

}

// 将链表合并
func mergeList_148(left, right *ListNode) *ListNode {
	//	 采用虚拟头结点进行计算
	dumy := &ListNode{Val: 0}
	pre := dumy
	for left != nil && right != nil {
		if left.Val < right.Val {
			pre.Next = left
			left = left.Next
		} else {
			pre.Next = right
			right = right.Next
		}
		pre = pre.Next
	}
	if left != nil {
		pre.Next = left
	}
	if right != nil {
		pre.Next = right
	}
	return dumy.Next
}

// 148 链表排序
func sortList(head *ListNode) *ListNode {
	dumy := &ListNode{Val: 0}
	dumy.Next = head
	root := head
	for root != nil && root.Next != nil {
		// 当root的值小于它的下一个值 就进行递推
		if root.Val <= root.Next.Val {
			root = root.Next
		} else {
			//	 大于的时候
			temp := root.Next
			cur := dumy
			root.Next = root.Next.Next
			for cur.Next.Val <= temp.Val {
				cur = cur.Next
			}
			temp.Next = cur.Next
			cur.Next = temp
		}
	}
	return dumy.Next
}

// 147 链表的插入排序
func insertionSortList(head *ListNode) *ListNode {
	dumyHead := &ListNode{Val: 0}
	dumyHead.Next = head
	if head == nil {
		return head
	}
	root := head
	for root != nil && root.Next != nil {
		if root.Val <= root.Next.Val {
			root = root.Next
		} else {
			//	 保存节点
			temp := root.Next
			//	 删除节点
			root.Next = root.Next.Next
			cur := dumyHead
			for temp.Val >= cur.Next.Val {
				cur = cur.Next
			}
			temp.Next = cur.Next
			cur.Next = temp

		}

	}

	return dumyHead.Next
}

// 145  二叉树的后序遍历  递归效率比非递归空间复杂度底
func postorderTraversal(root *TreeNode) []int {
	arr := []int{}
	if root == nil {
		return arr
	}
	// 递归
	//postorderTraversal_left(root, &arr)
	// 非递归
	stack := []*TreeNode{root}

	for len(stack) != 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		arr = append([]int{node.Val}, arr...)
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
	}

	return arr
}

func postorderTraversal_left(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}

	postorderTraversal_left(root.Left, arr)
	postorderTraversal_left(root.Right, arr)
	*arr = append(*arr, root.Val)
}

//  144 二叉树的前序遍历 非递归 递归比非递归空间复杂度高
func preorderTraversal(root *TreeNode) []int {
	arr := []int{}
	if root == nil {
		return arr
	}
	// 递归的方式
	//preorderTraversal_left(root, &arr)
	// 非递归的方式
	stack := []*TreeNode{root}
	for len(stack) != 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		arr = append(arr, node.Val)
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
	}
	return arr
}

//  144 二叉树的前序遍历递归
func preorderTraversal_left(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	*arr = append(*arr, root.Val)
	preorderTraversal_left(root.Left, arr)
	preorderTraversal_left(root.Right, arr)

}

// 143. 重排链表
func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}

	// 采用双指针判断链表的位置
	left_node := head
	right_node := head
	for right_node != nil {
		if right_node.Next == nil {
			break
		}
		left_node = left_node.Next
		right_node = right_node.Next.Next

	}
	//	将链表的后半段反转
	// 创建一个新的链表 表示反转链表
	var cur *ListNode
	// 创建新的链表
	root := left_node
	for root != nil {
		par := root.Next
		root.Next = cur
		cur = root
		root = par
	}

	//	 然后遍历head链表，每遍历一个链表 将反转链表的头部插入，
	//	base case 链表的两个节点相同的时候 终止
	//	将head链表当前节点之后置为空
	next := head
	for next != nil {
		par := next.Next
		next.Next = cur
		if cur == nil {
			break
		}
		cur = cur.Next
		next.Next.Next = par
		next = par
	}
	for head != nil {
		fmt.Println(head.Val)
		head = head.Next
	}
}

// 环形链表 II
func detectCycle(head *ListNode) *ListNode {
	//	 开始为-1
	one := head
	two := head
	for two != nil {
		if two.Next == nil {
			return nil
		}
		one = one.Next
		two = two.Next.Next
		if one == two {
			//	 当等于的时候 表示有环，接下来一人走一步
			three := head
			for {
				if two == three {
					return three
				}
				three = three.Next
				two = two.Next
			}
		}
	}
	return nil
}

//  141 环形链表
func hasCycle(head *ListNode) bool {
	// 选择两个节点 一个一次跳一下  一个一次跳两下
	// 当相等的时候 直接返回
	one := head
	two := head
	for two != nil {
		if two.Next == nil {
			return false
		}

		one = one.Next
		two = two.Next.Next
		if one == two {
			return true
		}

	}
	return false
}

// 140. 单词拆分 II
func wordBreak_140(s string, wordDict []string) []string {
	s_map := make(map[string]bool)
	//	 采用回溯算法
	for _, value := range wordDict {
		s_map[value] = true
	}
	arrs := []string{}
	wordBreak_140_dfs([]string{}, &arrs, 0, s_map, s)
	return arrs

}

func wordBreak_140_dfs(path []string, arrs *[]string, start int, s_map map[string]bool, s string) {
	//	 basecase
	if len(s) == start {
		cur := make([]string, len(path))
		copy(cur, path)
		str := path[0]
		for i := 1; i < len(path); i++ {
			str = str + " " + path[i]
		}
		*arrs = append(*arrs, str)
		return
	}

	for i := start + 1; i <= len(s); i++ {
		str := s[start:i]
		// 表示数据在这里面
		if s_map[str] {
			path = append(path, str)
			wordBreak_140_dfs(path, arrs, i, s_map, s)
			path = path[:len(path)-1]
		}
	}
}

// 139 单词拆分
func wordBreak(s string, wordDict []string) bool {
	s_map := make(map[string]bool)
	for _, value := range wordDict {
		s_map[value] = true
	}
	//	 创建dp序列
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for i := 1; i <= len(s); i++ {
		for j := i - 1; j >= 0; j-- {
			// 每次都重新输入数据
			str := s[j:i]
			// 因为是被空格拆分，所以这里就是出现可以拆分的单词 直接进行下一个处理
			if s_map[str] && dp[j] {
				// 这里dp[i]是因为str 不会包括i所以就是当前单词的下一个字符为true
				// 如果是最后一个字符，那么就是最后一个字符就是dp的最后一点为true
				dp[i] = true
				break
			}
		}
	}
	return dp[len(s)]
}

func copyRandomList(head *Node) *Node {
	node_map := make(map[*Node]*Node)
	cur := head
	for cur != nil {
		// 新建一个数据
		node_map[cur] = &Node{cur.Val, nil, nil}
		cur = cur.Next
	}

	for k, value := range node_map {
		value.Next = node_map[k.Next]
		value.Random = node_map[k.Random]
	}
	return node_map[head]

}

// 138 复制带有随机链表的指针
func copyRandomList_138(head *Node) *Node {
	if head == nil {
		return head
	}
	root := head
	// 先将后面的
	for root != nil {
		cur := &Node{root.Val, nil, nil}
		cur.Next = root.Next
		cur.Random = root.Random
		root.Next = cur
		root = cur.Next
	}
	// 进行指针的替换
	cur := head
	for cur != nil {
		if cur.Random != nil {
			par := cur.Next
			par.Random = cur.Random.Next
		}
		cur = cur.Next.Next
	}
	// 清除多余的链表
	ans := head.Next
	dumy := head
	for dumy != nil {
		next := dumy.Next
		if dumy.Next != nil {
			dumy.Next = dumy.Next.Next
		}
		dumy = next
	}

	return ans
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
