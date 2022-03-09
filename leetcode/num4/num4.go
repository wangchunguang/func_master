package num4

import (
	"fmt"
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
	Val       int
	Neighbors []*Node
}

func AddListNode() *ListNode {
	//head := &ListNode{1, &ListNode{2, &ListNode{3,&ListNode{4,&ListNode{5,nil}}}}}
	head := &ListNode{1, &ListNode{2, &ListNode{3,
		&ListNode{4, &ListNode{5, &ListNode{6,
			&ListNode{7, nil}}}}}}}
	return head
}

func AddTreeNode() *TreeNode {
	return &TreeNode{4,
		//&TreeNode{9, &TreeNode{23, nil, nil}, &TreeNode{8, nil, nil}},
		&TreeNode{9, &TreeNode{4, nil, nil}, &TreeNode{1, nil, nil}},
		&TreeNode{0, &TreeNode{0, nil, nil}, &TreeNode{0, nil, nil}}}
	//return &TreeNode{1,
	//	&TreeNode{2, nil, &TreeNode{3, nil,
	//		&TreeNode{4, nil, &TreeNode{5, nil, nil}}}},
	//	nil}

}

func AddArrNode() []*Node {
	return []*Node{
		&Node{1, []*Node{&Node{2, nil}, &Node{4, nil}}},
		&Node{2, []*Node{&Node{1, nil}, &Node{3, nil}}},
		&Node{3, []*Node{&Node{2, nil}, &Node{3, nil}}},
		&Node{4, []*Node{&Node{1, nil}, &Node{3, nil}}},
	}
}

// 并查集的实体
type UF struct {
	union []int
}

// 并查集数组里面包含的是当前数的上一个节点，开始默认的是包含自己本身
// 初始化并查集
func newUF(cap int) *UF {
	uf := UF{
		make([]int, cap),
	}
	for i := 0; i < cap; i++ {
		uf.union[i] = i
	}
	return &uf
}

// 有序表将两个单独的团队合并
func (u *UF) Union(x, y int) {
	rootX := u.find(x)
	rootY := u.find(y)
	if rootY == rootX {
		return
	}
	// 将y的根节点变为x的上一个节点
	u.union[rootX] = rootY
}

// 判断两个是不是同一个根节点
func (u *UF) Connected(x, y int) bool {
	return u.find(x) == u.find(y)
}

// 获取根结点
func (u *UF) find(x int) int {
	root := x
	// 获取根节点 ，因为根节点的指向的节点是它自己
	for u.union[root] != root {
		root = u.union[root]
	}
	// 如果开始查找根节点需要很多步，那么就替换当前节点的上一个节点为根节点
	for x != root {
		tmp := u.union[x]
		// 将查找节点所有的上序节点都改为指向根节点
		u.union[x] = root
		x = tmp
	}
	return root
}

type RNode struct {
	Next   *RNode
	random *RNode
}

// 136. 只出现一次的数字
func singleNumber(nums []int) int {
	num_map := make(map[int]int)
	// 设置初次出现的数据
	num := 0
	for i := 0; i < len(nums); i++ {
		if value, ok := num_map[nums[i]]; ok {
			num_map[nums[i]] = value + 1
		} else {
			num_map[nums[i]] = 1
		}

	}
	for k, v := range num_map {
		if v == 1 {
			num = k
			break
		}
	}
	return num
}

// 135分发糖果
func candy(ratings []int) int {
	dp := make([]int, len(ratings))
	// 先将每个人都分发一个糖果
	for i := 0; i < len(ratings); i++ {
		dp[i] = 1
	}
	sum := 0
	// 从前到后
	for i := 0; i < len(ratings)-1; i++ {
		// 前面的孩子比后面的孩子小
		if ratings[i] < ratings[i+1] {
			dp[i+1] = dp[i] + 1
		}
	}

	// 从后到前
	for i := len(ratings) - 1; i > 0; i-- {
		if ratings[i-1] > ratings[i] {
			// 从后面向前面的时候，如果前面的比后面的大，那么就在后面的基础上加1 和前面初始的糖果对比
			dp[i-1] = max(dp[i-1], dp[i]+1)
		}
	}

	for i := 0; i < len(ratings); i++ {
		sum += dp[i]
	}
	return sum
}

// 134 加油站
func canCompleteCircuit(gas []int, cost []int) int {
	lenght := len(gas)
	// 余量
	left := 0
	// 开始的起点
	start := 0
	tota := 0
	for i := 0; i < lenght; i++ {
		tota += gas[i] - cost[i]
		//	 计算每一次的剩余量
		left += gas[i] - cost[i]
		if left < 0 {
			start = i + 1
			left = 0
		}
	}
	//	 如果总的油量不够 就不能解决 这里避免的就是最后一个数为起点的时候
	if tota < 0 {
		return -1
	}
	return start
}

// 133 克隆图
func cloneGraph(node *Node) *Node {
	if node == nil {
		return node
	}
	node_map := make(map[*Node]*Node, 0)
	stack := []*Node{node}
	for len(stack) != 0 {
		// 出栈
		value := stack[len(stack)-1]
		// 更新栈
		stack = stack[:len(stack)-1]
		// 当前值不在map的时候
		if _, ok := node_map[value]; !ok {
			newNode := new(Node)
			// 创建一个新的node
			newNode.Val = value.Val
			node_map[value] = newNode
		}
		for _, val := range value.Neighbors {
			if _, ok := node_map[val]; !ok {
				stack = append(stack, val)
			}
		}
	}
	for k, v := range node_map {
		for _, cv := range k.Neighbors {
			v.Neighbors = append(v.Neighbors, node_map[cv])
		}
	}

	return node_map[node]
	//return cloneGraph_dfs(node, node_map)
}

// 克隆图的递归算法
func cloneGraph_dfs(node *Node, node_map map[*Node]*Node) *Node {
	if node == nil {
		return node
	}
	// 判断map里面有没有数据
	if value, ok := node_map[node]; ok {
		return value
	}
	// 将数据初始化
	r := &Node{Val: node.Val, Neighbors: make([]*Node, len(node.Neighbors))}
	// 判断当前节点是否添加
	node_map[node] = r
	for i := 0; i < len(node.Neighbors); i++ {
		r.Neighbors[i] = cloneGraph_dfs(node.Neighbors[i], node_map)
	}
	return r

}

// 132 分割回文串
func minCut(s string) int {
	ispail := make([][]bool, len(s))
	for i := 0; i < len(s); i++ {
		ispail[i] = make([]bool, len(s))
	}
	// j为行 i为列
	for i := 0; i < len(s); i++ {
		for j := 0; j <= i; j++ {
			if i == j {
				ispail[j][i] = true
			} else if i-j == 1 && s[i] == s[j] {
				ispail[j][i] = true
				//	 因为当前是通过上一层推导出来的 ，大于两个数之后
				// 比如abba 因为abba中 比如代表的位置 i  i+1 .. j-1 j s[i] == s[j] 并且斜下方的是  是 i+1 到j-1的位置
				// 所以如果 i+1 并且j-1的为回文 并且s[i] == s[j]  那么 这个就是回文
			} else if i-j > 1 && s[i] == s[j] && ispail[j+1][i-1] == true {
				ispail[j][i] = true
			}
		}
	}
	fmt.Println(ispail)
	// 第二次dp
	dp := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		dp[i] = i
	}
	// 因为计算的是0-i的 如果选择一个切入点j，就分为0-j 和 j+1-i
	// i的列的意思 j是行的意思
	for i := 1; i < len(s); i++ {
		if ispail[0][i] {
			dp[i] = 0
			continue
		}
		// 采用j去划分 dp 每一次都是重新计算
		for j := 0; j < i; j++ {
			if ispail[j+1][i] { //  如果j+1 到i时回文
				// 对比两个参数 因为i位置每一次都会重头开始计算一次
				if dp[j]+1 < dp[i] {
					dp[i] = dp[j] + 1
				}
			}
		}
	}
	return dp[len(s)-1]
}

// 131采用bp方向解决问题
func partition_dp(s string) [][]string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(s); i++ {
		dp[i] = make([]bool, len(s))
	}
	// j为行 i为列
	for i := 0; i < len(s); i++ {
		for j := 0; j <= i; j++ {
			if i == j {
				dp[j][i] = true
			} else if i-j == 1 && s[i] == s[j] {
				dp[j][i] = true
				//	 因为当前是通过上一层推导出来的 ，大于两个数之后
				// 比如abba 因为abba中 比如代表的位置 i  i+1 .. j-1 j s[i] == s[j] 并且斜下方的是  是 i+1 到j-1的位置
				// 所以如果 i+1 并且j-1的为回文 并且s[i] == s[j]  那么 这个就是回文
			} else if i-j > 1 && s[i] == s[j] && dp[j+1][i-1] == true {
				dp[j][i] = true
			}
		}
	}
	fmt.Println(dp)
	arrs := [][]string{}
	partition_dp_dfs([]string{}, 0, &arrs, s, dp)
	return arrs
}

func partition_dp_dfs(path []string, start int, arrs *[][]string, s string, dp [][]bool) {
	if start == len(s) {
		cur := make([]string, len(path))
		copy(cur, path)
		*arrs = append(*arrs, cur)
		return
	}
	for i := start; i < len(s); i++ {
		if dp[start][i] {
			path = append(path, s[start:i+1])
			partition_dp_dfs(path, i+1, arrs, s, dp)
			path = path[:len(path)-1]
		}
	}
}

// 131 分割回文子串
func partition(s string) [][]string {

	arrs := [][]string{}
	// 数组的长度为
	memo := make([][]int, len(s))
	for i := 0; i < len(memo); i++ {
		memo[i] = make([]int, len(s))
	}

	partition_dfs(s, &arrs, []string{}, 0, memo)
	return arrs
}

// 采用回溯算法
func partition_dfs(s string, arrs *[][]string, path []string, start int, memo [][]int) {
	// base case 判断是不是回文字符串 为1的时候 回文就是他本身
	if len(s) == start {
		cur := make([]string, len(path))
		copy(cur, path)
		*arrs = append(*arrs, cur)
		return
	}
	for i := start; i < len(s); i++ {
		// 记录的信息不是回文子串
		if memo[start][i] == 2 {
			continue
		}
		if memo[start][i] == 1 {
			path = append(path, s[start:i+1])
			partition_dfs(s, arrs, path, i+1, memo)
			path = path[:len(path)-1]
		} else if is_palindrome(s, start, i, memo) {
			path = append(path, s[start:i+1])
			partition_dfs(s, arrs, path, i+1, memo)
			path = path[:len(path)-1]
		}
	}
}

// 回文
func is_palindrome(s string, l, r int, memo [][]int) bool {
	for l < r {
		if s[l] != s[r] {
			memo[l][r] = 2
			return false
		}
		l++
		r--
	}
	memo[l][r] = 1
	return true
}

//130 并查集 解法
func solve_130(board [][]byte) {
	m, n := len(board), len(board[0])
	if m <= 2 {
		return
	}
	//	 将二维数组转化成一位数组
	anchor := m * n
	// 加一是因为有0
	u := newUF(anchor + 1)
	// 构建连同的区域
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// 如果判断当前是‘0’
			if board[i][j] == 'O' {
				//	 获取在一维表中的数据
				curNum := genNum(i, j, n)
				// 如果是最后一行的
				if i == 0 || j == 0 || i == m-1 || j == n-1 {
					// 所有为0的 并且边界为0的都设置 根节点为一维表最后一个数据
					// 并查集设置为一个根节点
					u.Union(curNum, anchor)
				}
				//	 每次只判断左边 上边的元素 就可以覆盖全部
				// 每次判断左  上 也要对边界进行判断 不然也会漏掉
				// 上边为0
				if i-1 >= 0 && board[i-1][j] == 'O' {
					// 当前的数据 将它的上一个节点换为上边的节点
					u.Union(curNum, genNum(i-1, j, n))
				}
				//	 左边的数据为
				if j-1 >= 0 && board[i][j-1] == 'O' {
					u.Union(curNum, genNum(i, j-1, n))
				}
			}
		}
	}
	//	 判定连同区域
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// 判定值和指定的根节点是否是一个值，是的话就是连接的 不是的话 就不是连接的
			if board[i][j] == 'O' && !u.Connected(anchor, genNum(i, j, n)) {
				board[i][j] = 'X'
			}
		}
	}

}

func genNum(i, j, cols int) int {
	return i*cols + j
}

// 130 被围绕的区域
func solve(board [][]byte) {
	if len(board) < 3 || len(board[0]) < 3 {
		return // 这个矩阵最小也得是3x3的，否则不可能出现被包围的O
	}
	r, c := len(board), len(board[0]) // r -> row, c -> column
	for i := 0; i < r; i++ {          // 从第一行到最后行
		dfs(board, r, c, i, 0)   // 把每行首列的每个元素都进行一番祖宗十八辈的查找
		dfs(board, r, c, i, c-1) // 把最后一列的每个元素都进行一番祖宗十八辈的查找
	}
	for i := 1; i < c-1; i++ { // 从第二列到倒数第二列
		dfs(board, r, c, 0, i)   // 把第一行的每个元素都进行一番祖宗十八辈的查找
		dfs(board, r, c, r-1, i) // 把最后一行的每个元素都进行一番祖宗十八辈的查找
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ { // 无脑迭代每个元素
			switch board[i][j] {
			case 'A':
				board[i][j] = 'O' // 看见这个元素是A，那就把它变成O，因为它和边界的O相连
			case 'O':
				board[i][j] = 'X' // 看见这个元素是O，那就把它变成X，因为它冰清玉洁不和边界相连
			}
		}
	}
}

func dfs(board [][]byte, r, c, x, y int) {
	if x < 0 || x >= r || y < 0 || y >= c || board[x][y] != 'O' {
		return // 查找祖宗十八辈的时候发现越界了，或者这个元素压根就不是O，那它出身良好
	}
	board[x][y] = 'A' // 哎呀，竟然没越界并且这个元素是O，那就把你变成A，
	// 因为我们查找的时候，起始元素都是从边界开始的，能查到你说明你和边界的O有瓜葛，出身不行！
	dfs(board, r, c, x-1, y) // 查你上边一辈
	dfs(board, r, c, x+1, y) // 查你下边一辈
	dfs(board, r, c, x, y-1) // 查你左边一辈
	dfs(board, r, c, x, y+1) // 查你右边一辈
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
