package num3

import (
	"fmt"
	"math"
	"math/big"
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
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
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

// 129 求根节点到叶子节点之和
func sumNumbers(root *TreeNode) int {
	return sumNumbers_root(root, 0)
}

func sumNumbers_root(root *TreeNode, num int) int {
	if root == nil {
		return 0
	}
	// 因为父节点和子节点是累加的，所以是10倍
	num = 10*num + root.Val
	// 当左右子节点都没有了的时候 将数据返回
	if root.Left == nil && root.Right == nil {
		return num
	}
	// 左子树和右子树的和
	return sumNumbers_root(root.Left, num) + sumNumbers_root(root.Right, num)
}

// 128最长连续序列
func longestConsecutive(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	num_map := make(map[int]bool)
	for _, value := range nums {
		num_map[value] = true
	}
	max_num := 0

	for key, _ := range num_map {
		// 如果有 key-1的数不需要比较，因为表示已经比较过 只需要比较向右的数据
		if num_map[key-1] {
			continue
		}
		tmpCount := 0
		for num_map[key] {
			key++
			tmpCount++
		}
		if max_num < tmpCount {
			max_num = tmpCount
		}

	}
	return max_num
}

// 127 单词接龙
func ladderLength(beginWord string, endWord string, wordList []string) int {
	wordMap := map[string]bool{}
	for _, w := range wordList {
		wordMap[w] = true
	}
	// 采用bfs 就用队列的形式
	queue := []string{beginWord}
	level := 1
	for len(queue) != 0 {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			word := queue[0]
			queue = queue[1:]
			if word == endWord {
				return level
			}
			// 遍历当前单词的所有字符
			for c := 0; c < len(word); c++ {
				for j := 'a'; j <= 'z'; j++ { //对应26个字符，每一个字符进行新增其中一个，然后去对比所有的字符，
					newWord := word[:c] + string(j) + word[c+1:]
					// 将符合条件的 设置为false
					if wordMap[newWord] == true {
						queue = append(queue, newWord)
						wordMap[newWord] = false
					}
				}
			}
		}
		level++
	}
	return 0
}

// 126 单词接龙2
func findLadders(beginWord string, endWord string, wordList []string) [][]string {
	lenght := len(wordList)
	if lenght == 0 {
		return nil
	}
	// 判断单词是否出现过
	w_map := map[string]bool{}
	for _, value := range wordList {
		w_map[value] = true
	}
	// 判断当前单词是否结尾
	flag := false
	// 用于进行筛选的数组
	queue_arr := [][]string{}
	// 用于返回的数组
	list_arr := [][]string{}
	// 因为采用广度优先，所以会采用队列的形式表示所有符合条件的数据
	queue := []string{beginWord}
	// 创建的数组 存储符合条件的要求
	arr := []string{}
	arr = append(arr)
	for len(queue) != 0 {
		if flag {
			return list_arr
		}
		listSize := len(queue)
		// 采用的广度优先算法 采用队列的形式
		for i := 0; i < listSize; i++ {
			word := queue[0]
			queue = queue[1:]
			if word == endWord {
				flag = true
				// 将数组添加到返回的数组
				arr = append(arr, word)
				// 已经完成之后，将数组数组返回，
				arr = arr[:len(arr)-1]
				break
			}
			// flag = true 表示里面有最短路径了，所以不需要继续做后面的处理
			if flag {
				break
			}
			// 判断可以符合条件的进行添加到数组
			for c := 0; c < len(word); c++ {
				for j := 'a'; j <= 'z'; j++ {
					newWord := word[:c] + string(j) + word[c+1:]
					if w_map[newWord] == true {
						// 当前符合要求的
						queue = append(queue, newWord)
						queue_arr = append(queue_arr, queue)
						w_map[newWord] = false
					}
				}
			}
		}
	}
	fmt.Println(queue_arr)
	return list_arr
}

// 125 验证是否是回文字符串
func isPalindrome(s string) bool {
	n := len(s)
	str := ""
	for i := 0; i < n; i++ {
		if s[i] >= 97 && s[i] <= 122 {
			str += string(s[i])
		} else if s[i] >= 65 && s[i] <= 90 {
			str += string(s[i] + 32)
		} else if s[i] >= 48 && s[i] <= 57 {
			str += string(s[i])
		}
	}
	num := len(str)
	if num < 2 {
		return true
	}
	age := num / 2
	for i := 0; i < age; i++ {
		if str[i] == str[num-1] {
			num--
		} else {
			return false
		}
	}
	return true
}

// 124 二叉树的最大路径
func maxPathSum(root *TreeNode) int {
	var maxPathSum_dfs func(root *TreeNode) int
	maxsum := math.MinInt64
	maxPathSum_dfs = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		left := maxPathSum_dfs(root.Left)
		right := maxPathSum_dfs(root.Right)

		maxsum = max(root.Val+left+right, maxsum)
		// 在左右两个子节点中选择较大的一个和当前的相加
		sum := root.Val + max(left, right)
		return max(sum, 0)
	}
	maxPathSum_dfs(root)
	return maxsum
}

//  买卖股票的最佳时机 III
func maxProfit3(prices []int) int {
	// 分为六种状态
	// dp[天数][表示状态]
	// dp[i][0] 没有操作
	// dp[i][1] 第一次买入
	// dp[i][2] 第一次卖出
	// dp[i][3] 第二次买入
	// dp[i][4] 第二次卖出
	n := len(prices)
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 5)
	}
	// 第一天的信息
	// 不操作
	dp[0][0] = 0
	// 第一次买入
	dp[0][1] = -prices[0]
	// 第一次卖出
	dp[0][2] = 0
	// 第二次买入
	dp[0][3] = -prices[0]
	// 第二次卖出
	dp[0][4] = 0
	for i := 1; i < n; i++ {
		dp[i][0] = dp[i-1][0]
		// 第一次买入可能是当天买入，也可能是以前买入
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
		// 第一次卖出可能是以前卖出 也可能是当前卖出
		dp[i][2] = max(dp[i-1][2], dp[i-1][1]+prices[i])
		// 第二次买入 可能是当天买入 也可能是以前买入
		dp[i][3] = max(dp[i-1][3], dp[i-1][2]-prices[i])
		// 第二次卖出，可能是当天卖出 也可能是以前卖出
		dp[i][4] = max(dp[i-1][4], dp[i-1][3]+prices[i])
	}

	return dp[n-1][4]
}

// 122. 买卖股票的最佳时机 II
func maxProfit2(prices []int) int {
	num := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			num += prices[i] - prices[i-1]
		}
	}
	return num
}

// 动态规划实现
func maxProfit_122(prices []int) int {
	lenght := len(prices)
	if lenght < 2 {
		return 0
	}
	dp := make([][]int, lenght)
	for i := 0; i < lenght; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = 0
	dp[0][1] = -prices[0]
	for i := 1; i < lenght; i++ {
		// 当前不持有股票 表示前一天不持有补票或者前一天持有股票当天卖出
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		//	当天持有股票  表示前一天持有股票，或者前一天不持有股票 当天买入
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
	}
	return dp[lenght-1][0]
}

// 121 买卖股票的最佳时机
func maxProfit(prices []int) int {
	//	 前i天最小的价格
	min_num := prices[0]
	num := 0
	for i := 1; i < len(prices); i++ {
		// 获取出来前i个数中的最小值
		min_num = min(min_num, prices[i])
		num = max(num, prices[i]-min_num)
	}
	return num
}

func maxProfit_121(prices []int) int {
	lenght := len(prices)
	if lenght < 2 {
		return 0
	}
	dp := make([][]int, lenght)
	for i := 0; i < lenght; i++ {
		dp[i] = make([]int, 2)
	}
	// 表示当天不持有股票
	dp[0][0] = 0
	// 表示当天持有股票
	dp[0][1] = -prices[0]
	for i := 1; i < lenght; i++ {
		// 表示当天不持有股票，1.前一天不持有股票， 2.前一天持有股票，当天卖出股票
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		// 当天持有股票，前一天持有股票，或者前一天不持有股票，当天买入股票
		dp[i][1] = max(dp[i-1][1], -prices[i])
	}
	return dp[lenght-1][0]
}

// 120 三角形最小路径和
func minimumTotal(triangle [][]int) int {
	// 初始化dp
	dp := make([][]int, len(triangle))
	for i := 0; i < len(triangle); i++ {
		dp[i] = make([]int, len(triangle[i]))
	}
	dp[0][0] = triangle[0][0]

	for i := 1; i < len(triangle); i++ {
		for j := 0; j < len(triangle[i]); j++ {
			// 判断j是否等于0的时候
			if j == 0 {
				dp[i][j] = dp[i-1][j] + triangle[i][j]
			} else if j == len(triangle[i])-1 {
				dp[i][j] = dp[i-1][j-1] + triangle[i][j]
			} else {
				dp[i][j] = triangle[i][j] + min(dp[i-1][j-1], dp[i-1][j])
			}
		}
	}
	arr := dp[len(triangle)-1]
	num := math.MaxInt64
	for i := 0; i < len(arr); i++ {
		num = min(num, arr[i])
	}
	return num
}

// 119 杨辉三角2
func getRow(rowIndex int) []int {
	dp := make([][]int, rowIndex+1)
	for i := 0; i <= rowIndex; i++ {
		dp[i] = make([]int, i+1)
		dp[i][0] = 1
		dp[i][i] = 1
	}
	if rowIndex < 2 {
		return dp[rowIndex]
	}
	for i := 2; i <= rowIndex; i++ {
		for j := 1; j < i; j++ {
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
		}
	}
	return dp[rowIndex]
}

// 118 杨辉三角
func generate(numRows int) [][]int {
	dp := make([][]int, numRows)
	if numRows == 0 {
		return dp
	}
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, i+1)
		dp[i][0] = 1
		dp[i][i] = 1
	}
	if numRows < 2 {
		return dp
	}
	for i := 2; i < numRows; i++ {
		for j := 1; j < i; j++ {
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
		}
	}
	return dp
}

// 117 填充每个节点的下一个右侧节点指针 II
func connect2(root *Node) *Node {
	if root == nil {
		return nil
	}

	cur := root //指针
	for cur != nil {
		var pre *Node  //前置节点
		var down *Node //下降节点，节点为下一层的左边第一节点
		for cur != nil {
			if cur.Left != nil { //左节点判断
				if pre != nil {
					pre.Next = cur.Left //pre不为空 设置Next
				} else {
					down = cur.Left //pre为空 设置下降节点
				}
				pre = cur.Left //设置前置节点
			}

			if cur.Right != nil { //右节点判断，同上
				if pre != nil {
					pre.Next = cur.Right
				} else {
					down = cur.Right
				}
				pre = cur.Right
			}
			cur = cur.Next //同层移动
		}
		cur = down //下降
	}
	return root

}

// 116 ，填充每个节点的下一个右侧节点指针
func connect(root *Node) *Node {
	if root == nil {
		return nil
	}
	n := root
	var nextLevel *Node
	for n.Left != nil {
		nextLevel = n.Left
		for n != nil {
			n.Left.Next = n.Right
			if n.Next != nil {
				n.Right.Next = n.Next.Left
			}
			n = n.Next
		}
		n = nextLevel
	}
	return root
}

// 115. 不同子序列
func numDistinct(s string, t string) int {
	slen, tlen := len(s), len(t)
	dp := make([][]int, slen+1)
	for i := 0; i <= slen; i++ {
		dp[i] = make([]int, tlen+1)
		dp[i][0] = 1
	}

	for i := 1; i <= slen; i++ {
		for j := 1; j <= tlen; j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}
	return dp[slen][tlen]
}

// 114 二叉树展开链表
func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	arr := []int{}
	flatten_preface(root, &arr)
	node := root
	for i := 0; i < len(arr); i++ {
		root.Val = arr[i]
		root.Left = nil
		if root.Right == nil && i != len(arr)-1 {
			root.Right = &TreeNode{}
		}
		root = root.Right
	}
	root = node
}

// 采用先序遍历
func flatten_preface(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	*arr = append(*arr, root.Val)
	flatten_preface(root.Left, arr)
	flatten_preface(root.Right, arr)
}

//113 采用回溯算法
func pathSum_113(root *TreeNode, targetSum int) [][]int {
	array := [][]int{}
	if root == nil {
		return array
	}
	pathSum_113_dfs(root, root.Val, targetSum, &array, []int{root.Val})
	return array
}

func pathSum_113_dfs(root *TreeNode, sum, targetSum int, array *[][]int, arr []int) {
	if root == nil {
		return
	}
	// base case 条件是 没有来了根节点，而且值符合条件的情况下 ，将数组添加进二维数组
	if root.Left == nil && root.Right == nil {
		if sum == targetSum {
			// 将值添加进去
			tmp := make([]int, len(arr))
			copy(tmp, arr)
			*array = append(*array, tmp)
		}

		return
	}
	// 左边不等于空 回溯
	if root.Left != nil {
		arr = append(arr, root.Left.Val)
		sum = sum + root.Left.Val
		pathSum_113_dfs(root.Left, sum, targetSum, array, arr)
		sum = sum - root.Left.Val
		arr = arr[:len(arr)-1]

	}
	//	 右边不等于空 回溯
	if root.Right != nil {
		arr = append(arr, root.Right.Val)
		sum = sum + root.Right.Val
		pathSum_113_dfs(root.Right, sum, targetSum, array, arr)
		sum = sum - root.Right.Val
		arr = arr[:len(arr)-1]
	}
}

// 113 路径总和dsf 2
func pathSum(root *TreeNode, targetSum int) [][]int {
	array := [][]int{}
	if root == nil {
		return array
	}
	pathSum_dfs(root, targetSum, []int{}, &array)
	return array
}

func pathSum_dfs(root *TreeNode, targetSum int, arr []int, array *[][]int) {
	if root == nil {
		return
	}
	// 判断是否符合目标 采用前序遍历
	if root.Left == nil && root.Right == nil && root.Val == targetSum {
		arr = append(arr, root.Val)
		tmp := make([]int, len(arr))
		copy(tmp, arr)
		*array = append(*array, tmp)
		return
	}
	arr = append(arr, root.Val)
	pathSum_dfs(root.Left, targetSum-root.Val, arr, array)
	pathSum_dfs(root.Right, targetSum-root.Val, arr, array)

}

// 迭代方式 对每一个节点的值都存储，如果左右节点为kong 判断条件
func hasPathSum_112(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	stack := []*TreeNode{root}
	arr := []int{root.Val}
	for len(stack) != 0 {
		// 出队列
		node := stack[0]
		num := arr[0]
		stack = stack[1:]
		arr = arr[1:]
		if node.Left == nil && node.Right == nil {
			if targetSum == num {
				return true
			}
			// 不满足条件就直接跳过
			continue
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
			arr = append(arr, node.Left.Val+num)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
			arr = append(arr, node.Right.Val+num)
		}
	}
	return false
}

// 112 路径总和 递归方式
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	//	 当成为叶子结点的时候 判断是不满足要求
	if root.Left == nil && root.Right == nil {
		return targetSum == root.Val
	}
	// 不断更新左边的值
	l := hasPathSum(root.Left, targetSum-root.Val)
	r := hasPathSum(root.Right, targetSum-root.Val)
	return l || r
}

// 递归版本
func minDepth_111(root *TreeNode) int {
	if root == nil {
		return 0
	}
	ans := 0
	//	 左右字数都为空的情况
	if root.Left == nil && root.Right == nil {
		ans = 1
	} else if root.Left != nil && root.Right != nil { // 左子树都不为空
		ans = min(minDepth_111(root.Left), minDepth_111(root.Right)) + 1
	} else if root.Left != nil {
		ans = minDepth_111(root.Left) + 1
	} else {
		ans = minDepth_111(root.Right) + 1
	}
	return ans
}

// 111 求二叉树的最小深度
func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	count := 1
	endnode := root
	nextnode := root
	stack := []*TreeNode{}
	stack = append(stack, root)
	for len(stack) != 0 {
		node := stack[0]
		stack = stack[1:]
		if node.Left == nil && node.Right == nil {
			return count
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
			nextnode = node.Left
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
			nextnode = node.Right
		}
		if endnode == node {
			endnode = nextnode
			nextnode = nil
			count++
		}
	}
	return count
}

//110 判断一棵树是不是高度平衡的二叉树
func isBalanced(root *TreeNode) bool {
	return isBalanced_110(root) != -1
}

// 递归计算左子树和右子树的高度
func isBalanced_110(root *TreeNode) float64 {
	if root == nil {
		return 0
	}
	left := isBalanced_110(root.Left)
	if left == -1 {
		return -1
	}
	right := isBalanced_110(root.Right)
	if right == -1 {
		return -1
	}
	if math.Abs(left-right) > 1 {
		return -1
	}
	return math.Max(left, right) + 1
}

// 将有序链表转换为二叉树
func sortedListToBST_109(head *ListNode) *TreeNode {
	if head == nil {
		return nil
	}
	arr := []int{}
	for head != nil {
		arr = append(arr, head.Val)
		head = head.Next
	}
	return sortedArrayToBST_108_1(0, len(arr)-1, arr)

}

// 将有序数组转换为二叉树
func sortedArrayToBST_108(nums []int) *TreeNode {
	return sortedArrayToBST_108_1(0, len(nums)-1, nums)
}

func sortedArrayToBST_108_1(start, end int, nums []int) *TreeNode {

	if start > end {
		return nil
	}
	mid := start + (end-start)/2
	root := &TreeNode{nums[mid], nil, nil}
	root.Left = sortedArrayToBST_108_1(start, mid-1, nums)
	root.Right = sortedArrayToBST_108_1(mid+1, end, nums)
	return root
}

// 二叉树的层序遍历
func levelOrderBottom_107(root *TreeNode) [][]int {
	array := [][]int{}
	if root == nil {
		return array
	}
	// 记录当前层完成的节点
	rootendnext := root
	// 记录当前存放到队列的节点
	rootnext := root
	// 将数据存放到队列
	stack := []*TreeNode{}
	stack = append(stack, root)
	arr := []int{}
	for len(stack) != 0 {
		//	 从队列取出元素
		node := stack[0]
		stack = stack[1:]
		// 判断左子树是否为空
		if node.Left != nil {
			stack = append(stack, node.Left)
			rootnext = node.Left
		}
		// 判断右子树是否为空
		if node.Right != nil {
			stack = append(stack, node.Right)
			rootnext = node.Right
		}
		arr = append(arr, node.Val)
		//  当前层的节点已经走完了
		if rootendnext == node {
			// 更新新的头结点
			rootendnext = rootnext
			rootnext = nil
			array = append(array, arr)
			arr = arr[len(arr):]
		}
	}
	if len(arr) != 0 {
		array = append(array, arr)
	}
	arrs := [][]int{}
	for i := len(array) - 1; i >= 0; i-- {
		arrs = append(arrs, array[i])
	}

	return arrs
}

// 106从中序与后续遍历序列构造二叉树
func buildTree_106(inorder []int, postorder []int) *TreeNode {
	if len(inorder) == 0 || len(postorder) == 0 {
		return nil
	}
	// 后续遍历的最后一个是二叉树的根节点
	// 将插入最后的一个节点
	// 找到后续遍历找那个的最后一个数在中序遍历中的位置
	num := 0
	for key, value := range inorder {
		if value == postorder[len(postorder)-1] {
			num = key
			break
		}
	}
	root := &TreeNode{Val: postorder[len(postorder)-1]}
	// 左子树的要求为
	root.Left = buildTree_106(inorder[:num], postorder[:num])
	root.Right = buildTree_106(inorder[num+1:], postorder[num:len(postorder)-1])

	return root
}

// 105 从前序与中序遍历序列构造二叉树
func buildTree_105(preorder []int, inorder []int) *TreeNode {
	// 如果有一个为空 直接返回
	if len(preorder) == 0 || len(inorder) == 0 {
		return nil
	}
	// 因为前序遍历的第一个就是根节点
	// 然后在中序遍历找到该位置，表示左边为左子树 右边为右子树
	root := &TreeNode{Val: preorder[0]}
	num := 0
	for kev, value := range inorder {
		if value == preorder[0] {
			num = kev
			break
		}
	}
	// 生成左子树 preorder因为第一个成为了头结点 因为切片的后面不包括 所以是+1
	root.Left = buildTree_105(preorder[1:num+1], inorder[:num])
	root.Right = buildTree_105(preorder[num+1:], inorder[num+1:])
	return root
}

func middle_order_buildTree(start, end int, preorder []int) *TreeNode {
	if start > end {
		return nil
	}
	mid := start + (end-start)/2
	root := &TreeNode{preorder[mid], nil, nil}
	root.Left = middle_order_buildTree(start, mid-1, preorder)
	root.Right = middle_order_buildTree(mid+1, end, preorder)

	return root
}

// 104 求二叉树的最大深度
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := maxDepth(root.Left)
	right := maxDepth(root.Right)
	return max(left, right) + 1
}

// 103 二叉树的锯齿形层遍历
func zigzagLevelOrder(root *TreeNode) [][]int {
	// 宽度有限遍历 采用队列的形式
	arrs := [][]int{}
	if root == nil {
		return arrs
	}
	//	表示当前节点最后的点
	var nodeCurend *TreeNode
	// 表示当前层数据的位置
	var nodenextend *TreeNode
	// 上一层为1
	nodeCurend = root
	//	创建一个队列创建
	roots := []*TreeNode{}
	roots = append(roots, root)
	floor := 1
	// 存储每一层的数组
	arr := []int{}
	for len(roots) != 0 {
		//	 出队列
		node := roots[0]
		// 更新队列
		roots = roots[1:]
		//	 先放左树 再放右树
		if node.Left != nil {
			roots = append(roots, node.Left)
			nodenextend = node.Left
		}
		if node.Right != nil {
			roots = append(roots, node.Right)
			nodenextend = node.Right
		}
		//	 如果上一层的点和当前出栈的点一样 表示需要更新数据
		if nodeCurend == node {
			// 更新下一层最新的节点
			nodeCurend = nodenextend
			nodenextend = nil
			arr = append(arr, node.Val)
			// 判断是奇数层还是偶数层 ，偶数层就反转代码
			if floor%2 == 1 {
				arrs = append(arrs, arr)
			} else {
				rra := []int{}
				for i := len(arr) - 1; i >= 0; i-- {
					rra = append(rra, arr[i])
				}
				arrs = append(arrs, rra)
				rra = rra[len(rra):]
			}

			arr = arr[len(arr):]
			floor++
		} else {
			arr = append(arr, node.Val)
		}
	}
	if len(arr) > 0 {
		if floor%2 == 1 {
			arrs = append(arrs, arr)
		} else {
			rra := []int{}
			for i := len(arr) - 1; i >= 0; i-- {
				rra = append(rra, arr[i])
			}
			arrs = append(arrs, rra)
			rra = rra[len(rra):]
		}
		arrs = append(arrs, arr)
	}
	return arrs
}

// 102二叉树的宽度遍历
func levelOrder1(root *TreeNode) [][]int {
	// 宽度有限遍历 采用队列的形式
	arrs := [][]int{}
	if root == nil {
		return arrs
	}
	//	表示当前节点最后的点
	var nodeCurend *TreeNode
	// 表示当前层数据的位置
	var nodenextend *TreeNode
	// 上一层为1
	nodeCurend = root
	//	创建一个队列创建
	roots := []*TreeNode{}
	roots = append(roots, root)
	// 存储每一层的数组
	arr := []int{}
	for len(roots) != 0 {
		//	 出队列
		node := roots[0]
		// 更新队列
		roots = roots[1:]
		//	 先放左树 再放右树
		if node.Left != nil {
			roots = append(roots, node.Left)
			nodenextend = node.Left
		}
		if node.Right != nil {
			roots = append(roots, node.Right)
			nodenextend = node.Right
		}
		//	 如果上一层的点和当前出队列的点一样 表示需要更新数据
		if nodeCurend == node {
			// 更新下一层最新的节点
			nodeCurend = nodenextend
			nodenextend = nil
			arr = append(arr, node.Val)
			arrs = append(arrs, arr)
			arr = arr[len(arr):]
		} else {
			arr = append(arr, node.Val)
		}
	}
	if len(arr) > 0 {
		arrs = append(arrs, arr)
	}

	return arrs
}

// 二叉树的宽度遍历 采用哈希表的形式
func levelOrder(root *TreeNode) [][]int {
	// 宽度有限遍历 采用队列的形式
	arrs := [][]int{}
	if root == nil {
		return arrs
	}
	m := make(map[*TreeNode]int)
	m[root] = 1
	//	创建一个队列创建
	roots := []*TreeNode{}
	// 将数据入队列
	roots = append(roots, root)
	// 当前第一层
	curlevel := 1

	arr := []int{}
	// 队列为空的话跳出循环
	for len(roots) != 0 {
		// 出队列
		node := roots[0]
		roots = roots[1:]
		// 当前层数
		num := m[node]
		// 如果层数一样的话 将数据添加到数组
		if num == curlevel {
			// 是同一层 将数据添加到同一层的数组
			arr = append(arr, node.Val)
		} else {
			//	 不是同一层
			arrs = append(arrs, arr)
			//	 将arr初始化
			arr = append(arr, node.Val)
			arr = arr[len(arr)-1:]
			curlevel++
		}
		// 有左子节点的时候
		if node.Left != nil {
			m[node.Left] = num + 1
			roots = append(roots, node.Left)
		}
		//	 有右子节点的时候
		if node.Right != nil {
			m[node.Right] = num + 1
			roots = append(roots, node.Right)
		}
	}
	arrs = append(arrs, arr)
	return arrs
}

// 对称二叉树
func isSymmetric(root *TreeNode) bool {
	left := root.Left
	right := root.Right
	return isSymmetric1(left, right)

}

func isSymmetric1(left, right *TreeNode) bool {
	if left == nil && right == nil {
		return true
	}
	if left == nil || right == nil {
		return false
	}
	if !isSymmetric1(left.Left, right.Left) {
		return false
	}
	fmt.Println(left.Val, right.Val)
	if left.Val != right.Val {
		return false
	}
	return isSymmetric1(left.Right, right.Right)
}

// 100 相同的树
func isSameTree(p *TreeNode, q *TreeNode) bool {
	return isSameTree1(p, q)
}

func isSameTree1(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	if !isSameTree1(p.Left, q.Left) {
		return false
	}
	if p.Val != q.Val {
		return false
	}

	return isSameTree1(p.Right, q.Right)

}

// 恢复二叉搜索树
func recoverTree(root *TreeNode) {
	// 先将二叉树通过中序遍历获取出来原数组 然后经过排序 然后再生成二叉树
	var arr []int
	recoverTree_BST(root, &arr)
	sort.Ints(arr)
	iterator := 0
	inorderFill(root, arr, &iterator)
}

func recoverTree_BST(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	recoverTree_BST(root.Left, arr)
	*arr = append(*arr, root.Val)
	recoverTree_BST(root.Right, arr)
}

func inorderFill(root *TreeNode, arr []int, current *int) {
	if root == nil {
		return
	}
	inorderFill(root.Left, arr, current)
	root.Val = arr[*current]
	*current++
	inorderFill(root.Right, arr, current)
}

// 中序遍历非递归
func middle_order(root *TreeNode) bool {
	if root == nil {
		return true
	}
	par := math.MinInt64
	//	 中序遍历，先将左右的左子树进栈，出栈的时候将自己的右子树进栈
	stack := []*TreeNode{}
	// 先将所有的左子树进栈
	for root != nil {
		stack = append(stack, root)
		root = root.Left
	}
	//arr := []int{}
	// 当左右的左子树进栈之后开始进行出栈操作
	for len(stack) != 0 {
		// 出栈
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		// 将数据添加到数组中
		//arr = append(arr, node.Val)
		if node.Val <= par {
			return false
		}
		par = node.Val
		node = node.Right
		for node != nil {
			stack = append(stack, node)
			node = node.Left
		}
	}
	return true
}

func isValidBST(root *TreeNode) bool {
	pre := math.MinInt64
	var helper func(root *TreeNode) bool
	helper = func(root *TreeNode) bool {
		if root == nil {
			return true
		}
		if !helper(root.Left) {
			return false
		}
		if root.Val <= pre {
			return false
		}
		pre = root.Val
		return helper(root.Right)
	}
	return helper(root)

}

// 97 交错字符串
func isInterleave(s1 string, s2 string, s3 string) bool {
	str1 := len(s1)
	str2 := len(s2)
	str3 := len(s3)
	if s1 == "" && s2 == "" && s3 == "" {
		return true
	}
	// 长度不一样 直接淘汰
	if str3 != (str2 + str1) {
		return false
	}

	// 动态规划解决问题
	// 假设s1 为行 s2位列
	dp := make([][]bool, str1+1)
	for i := 0; i < str1+1; i++ {
		dp[i] = make([]bool, str2+1)
	}
	dp[0][0] = true
	// 从第一行遍历
	for i := 0; i <= str1; i++ {
		for j := 0; j <= str2; j++ {
			// 计算s3的方式
			p := i + j - 1
			if i > 0 {
				dp[i][j] = dp[i][j] || dp[i-1][j] && s1[i-1] == s3[p]
			}
			if j > 0 {
				dp[i][j] = dp[i][j] || dp[i][j-1] && s2[j-1] == s3[p]
			}
		}
	}
	return dp[str1][str2]
}

// 构建二叉树
func createBinaryTree(n int) *TreeNode {
	return helper(1, n)
}

func helper(start, end int) *TreeNode {
	// 边界的条件
	if start > end {
		return nil
	}
	val := start + (end-start)/2
	root := &TreeNode{val, nil, nil}
	root.Left = helper(start, val-1)
	root.Right = helper(val+1, end)
	return root

}

// 生成不同的二叉树
func generateTrees(n int) []*TreeNode {
	if n == 0 {
		return nil
	}
	//	 采用回溯算法
	return generateTrees_backtrack(1, n)
}

// 生成二叉树
func generateTrees_backtrack(left, right int) []*TreeNode {
	//	 生成二叉树的方法
	if left > right {
		return []*TreeNode{nil}
	}
	roots := []*TreeNode{}
	//	 将每一个点都尝试作为根节点
	for i := left; i <= right; i++ {
		// 这里尝试了左右的树为节点的方法
		left_root := generateTrees_backtrack(left, i-1)
		right_root := generateTrees_backtrack(i+1, right)
		for m := 0; m < len(left_root); m++ {
			for n := 0; n < len(right_root); n++ {
				tree := &TreeNode{Val: i}
				tree.Left = left_root[m]
				tree.Right = right_root[n]
				roots = append(roots, tree)
			}
		}
	}
	return roots
}

// 采用栈的二叉树中序遍历
func stack_inorderTraversal(root *TreeNode) []int {
	arr := []int{}
	stack := []*TreeNode{}
	//	 中序遍历 就是先将所有的左子树全部入栈
	for root != nil {
		//arr = append(arr, root.Val)
		stack = append(stack, root)
		root = root.Left
	}
	// 进行出栈
	for len(stack) != 0 {
		// 栈的出栈
		node := stack[len(stack)-1]
		// 出栈后的数据
		stack = stack[:len(stack)-1]
		// 将数据添加进去
		stack = append(stack, node)
		node = node.Right
		for node != nil {
			arr = append(arr, node.Val)

			node = node.Left
		}

	}

	return arr
}

// 二叉树中序遍历
func inorderTraversal(root *TreeNode) []int {
	arr := []int{}
	middle_inorderTraversal(root, &arr)
	return arr
}

func middle_inorderTraversal(root *TreeNode, arr *[]int) {
	if root == nil {
		return
	}
	middle_inorderTraversal(root.Left, arr)
	*arr = append(*arr, root.Val)
	middle_inorderTraversal(root.Right, arr)
}

// 93 复原ip地址
func restoreIpAddresses(s string) []string {

	arr := []string{}
	restoreIpAddresses_backtrack([]string{}, 0, s, &arr)
	fmt.Println(arr)
	return arr
}

// 回溯三部曲
func restoreIpAddresses_backtrack(path []string, index int, s string, arr *[]string) {
	// 边界条件  数组里面是ip的四个数，并且长度走到了最后的一个
	if len(path) == 4 && len(s) == index {
		//	 满足条件 添加进指定的位置
		ip := path[0] + "." + path[1] + "." + path[2] + "." + path[3]
		*arr = append(*arr, ip)
		return
	}

	//	 循环
	for i := index; i < len(s); i++ {
		//	 添加进数组 数组的元素为 i-index
		// 因为切片是不包括最后一位，是最后一位减-1  所以需要i+1 才是 index到i+1
		path = append(path, s[index:i+1])
		// 判断递归条件
		// 每一个数的长度 不能超过三位
		// path长度不能超过四位
		// 如果当前的位数大于1位，那么首位不能是0 同时不能超过255
		if i-index+1 <= 3 && len(path) <= 4 && IsBoolIp(s, index, i) {
			//	 递归
			restoreIpAddresses_backtrack(path, i+1, s, arr)
		}
		//	回溯
		path = path[:len(path)-1]
	}

}

func IsBoolIp(s string, start, end int) bool {
	// 获取当前数的区间
	s2 := s[start : end+1]
	if end-start+1 > 1 && s[start] == '0' {
		return false
	}
	atoi, _ := strconv.Atoi(s2)
	if atoi > 255 {
		return false
	}
	return true
}

// 反转链表
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	// 旋转链表前面的部分
	left_list := new(ListNode)
	// 旋转链表的部分
	right_list := new(ListNode)
	l_list := left_list
	r_list := right_list
	index := 1
	// 先将旋转之前的链表提取出来
	for head != nil && index < left {
		left_list.Next = head
		head = head.Next
		left_list = left_list.Next
		index++
	}
	left_list.Next = nil
	for head != nil && index <= right {
		right_list.Next = head
		head = head.Next
		right_list = right_list.Next
		index++
	}
	right_list.Next = nil
	// 对于旋转的部分进行旋转
	r_list = r_list.Next
	var list *ListNode
	for r_list != nil {
		// 将循环的链表的下一个节点存储
		start := r_list.Next
		// 将循环的链表的下一个节点置为创建的节点
		r_list.Next = list
		// 将两个节点反转
		list = r_list
		r_list = start
	}
	left_list.Next = list
	for left_list.Next != nil {
		left_list = left_list.Next
	}
	left_list.Next = head

	return l_list.Next
}

// 解码方法
func numDecodings(s string) int {
	if len(s) == 0 || s[0] == '0' {
		return 0
	}
	dp := make([]int, len(s)+1)
	//	 第一位有数据
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= len(s); i++ {
		if s[i-1] != '0' {
			dp[i] = dp[i-1]
		}
		if s[i-2] == '1' || s[i-2] == '2' && s[i-1] < '7' {
			dp[i] = dp[i] + dp[i-2]
		}
	}
	fmt.Println(dp)
	return dp[len(s)]

}

// 子集2
func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	arr := [][]int{}
	if len(nums) == 0 {
		arr = append(arr, []int{})
		return arr
	}
	subsetsWithDup_backtrack(nums, []int{}, &arr, 0)
	return arr
}

// 回溯算法
func subsetsWithDup_backtrack(nums, path_arr []int, arr *[][]int, n int) {
	temp := make([]int, len(path_arr))
	copy(temp, path_arr)
	*arr = append(*arr, temp)
	for i := n; i < len(nums); i++ {
		// 因为排序过的，所以两个数相同就会直接跳过
		if i > n && nums[i] == nums[i-1] {
			continue
		}
		path_arr = append(path_arr, nums[i])
		subsetsWithDup_backtrack(nums, path_arr, arr, i+1)
		path_arr = path_arr[:len(path_arr)-1]
	}

}

func merge(nums1 []int, m int, nums2 []int, n int) {
	if len(nums1) < m {
		m = len(nums2)
	}
	if len(nums2) < n {
		n = len(nums2)
	}
	nums1 = nums1[:m]
	nums1 = append(nums1, nums2[:n]...)
	sort.Ints(nums1)
}

// 分割链表
func partition_86(head *ListNode, x int) *ListNode {
	min_list := &ListNode{}
	max_list := &ListNode{}
	currGreat := min_list
	currLess := max_list
	for head != nil {
		if head.Val < x {
			currLess.Next = head
			currLess = head
		} else {
			currGreat.Next = head
			currGreat = head
		}
		head = head.Next
	}
	currGreat.Next = nil
	currLess.Next = min_list.Next
	return max_list.Next
}

// 柱状图中最大的矩形
func largestRectangleArea(heights []int) int {
	//	 采用O（n²)的解法
	if len(heights) == 0 {
		return 0
	}
	lenght := len(heights)
	sum := heights[0]
	left_higt := make([]int, lenght)
	right_higt := make([]int, lenght)
	// 获取当前数据左边最大的数，没有就填写0
	for i := 1; i < lenght-1; i++ {
		//
		left_higt[i] = max(heights[i-1], left_higt[i-1])
	}

	// 获取当前数值右边最大的数没有就填写1
	for j := lenght - 2; j >= 0; j-- {
		right_higt[j] = max(heights[j+1], right_higt[j+1])
	}
	for i := 1; i < lenght-1; i++ {

	}
	return sum
}

// 删除链表的重复元素
func deleteDuplicates1(head *ListNode) *ListNode {
	cur := head
	for head != nil && head.Next != nil {
		if head.Val == head.Next.Val {
			// 循环删除
			for head.Next != nil && head.Val == head.Next.Val {
				head.Next = head.Next.Next
			}
		}
		head = head.Next
	}
	return cur
}

// 删除链表中的重复元素
func deleteDuplicates(head *ListNode) *ListNode {
	list := new(ListNode)
	list.Next = head
	prev := list
	// 当前链表的值
	for head != nil && head.Next != nil {
		//		 出现重复节点
		if head.Val == head.Next.Val {
			for head.Next != nil && head.Val == head.Next.Val {
				head.Next = head.Next.Next
			}
			prev.Next = head.Next
		} else {
			prev = prev.Next
		}
		head = head.Next

	}
	return list.Next
}

// 搜索旋转数组
func search(nums []int, target int) bool {
	// 采用二分查找算法
	right := len(nums) - 1
	left := 0
	for left <= right {
		mid := left + (right-left)/2
		//	 判断是否相等
		if nums[mid] == target {
			return true
		} else if nums[mid] == nums[left] && nums[mid] == nums[right] { //	 去除多余的数据
			left++
			right--

		} else if nums[mid] >= nums[left] { // 左边有序//	 因为是旋转的 所以判断是左边有序还是右边有序
			if nums[mid] > target && target >= nums[left] {
				right = mid - 1
			} else {
				left = mid + 1
			}

		} else if nums[mid] < nums[left] { // 右边有序
			if nums[mid] < target && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return false

}

// 搜索旋转数组
func search1(nums []int, target int) bool {
	if len(nums) == 0 {
		return false
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] == target {
			return true
		}
	}
	return false

}

func removeDuplicates(nums []int) int {
	var process func(k int) int
	process = func(k int) int {
		u := 0
		for _, v := range nums {
			// u < k 表示都符合条件  nums[u-k] != v当遇见钱k为不同的数据直接添加
			if u < k || nums[u-k] != v {
				nums[u] = v
				u++
			}
		}
		return u
	}
	return process(2)
}

// 单词搜索
func exist(board [][]byte, word string) bool {
	//	 多少行
	row_len := len(board)
	//  多少列
	list_len := len(board[0])
	// 创建矩阵判断字符是否用过
	used := make([][]bool, row_len)
	for i := 0; i < row_len; i++ {
		used[i] = make([]bool, list_len)
	}
	// 遍历行列
	for i := 0; i < row_len; i++ {
		for j := 0; j < list_len; j++ {
			//	 如果找到了起点，并且递归结果为真，就直接返回真
			if board[i][j] == word[0] && exist_backtrack(row_len, list_len, i, j, 0, used, word, board) {
				return true
			}
		}

	}
	return false
}

// 继续回溯算法
func exist_backtrack(row_len, list_len, row, list, num int, used [][]bool, word string, board [][]byte) bool {
	if num == len(word) {
		return true
	}
	//	 因为是上下左右，所以不能越界
	if row < 0 || row >= row_len || list < 0 || list >= list_len {
		return false
	}
	//	 当前节点已经访问过，或者就是非目标节点
	if used[row][list] || board[row][list] != word[num] {
		return false
	}
	//	 当排除了所有情况之后 ，当前节点暂时没有毛病可以继续访问
	used[row][list] = true
	// 前后左右判断,只要有一个满足条件都可以
	flag := exist_backtrack(row_len, list_len, row+1, list, num+1, used, word, board) ||
		exist_backtrack(row_len, list_len, row-1, list, num+1, used, word, board) ||
		exist_backtrack(row_len, list_len, row, list+1, num+1, used, word, board) ||
		exist_backtrack(row_len, list_len, row, list-1, num+1, used, word, board)

	if flag {
		return true
	} else {
		used[row][list] = false
		return false
	}
}

// 子集
func subsets(nums []int) [][]int {
	arr_list := make([][]int, 0)
	//	判断条件
	if len(nums) == 1 {
		arr_list = append(arr_list, []int{})
		arr_list = append(arr_list, nums)
		return arr_list
	}
	// 不重复的子集
	subsets_backtrack(nums, []int{}, &arr_list, 0)
	return arr_list
}

func subsets_backtrack(nums, arr []int, arr_list *[][]int, num int) {
	// 不是边界条件的时候，每个数组都添加进去
	tmp := make([]int, len(arr))
	copy(tmp, arr)
	*arr_list = append(*arr_list, tmp)
	for i := num; i < len(nums); i++ {
		arr = append(arr, nums[i])
		fmt.Println("入栈", arr, num)
		// +1是因为前面的数据后面循环的时候会直接跳过
		subsets_backtrack(nums, arr, arr_list, i+1)
		fmt.Println("出栈", arr, num)
		arr = arr[:len(arr)-1]
	}
}

// 组合
func combine1(n int, k int) [][]int {
	// 最后存储的数组
	arr_list := make([][]int, 0)
	if n <= 0 || k <= 0 || k > n {
		return arr_list
	}
	combine_backtrack1(n, k, 1, []int{}, &arr_list)
	fmt.Println(arr_list)
	return arr_list
}

func combine_backtrack1(n, k, start int, track []int, arr *[][]int) {
	if len(track) == k {
		tmp := make([]int, k)
		copy(tmp, track)
		*arr = append(*arr, tmp)
	}
	if len(track)+n-start+1 < k {
		return
	}
	for i := start; i <= n; i++ {
		track = append(track, i)
		fmt.Println("递归之前", track)
		combine_backtrack1(n, k, i+1, track, arr)
		fmt.Println("递归之前", track)
		track = track[:len(track)-1]
	}

}

// 组合
func combine(n int, k int) [][]int {
	// 最后存储的数组
	arr_list := make([][]int, 0)

	if n <= 0 || k <= 0 || k > n {
		return arr_list
	}
	combine_backtrack([]int{}, n, k, 1, &arr_list)
	return arr_list
}

func combine_backtrack(pathNum []int, n, k, start int, arr *[][]int) {
	if len(pathNum) == k {
		tmp := make([]int, k)
		copy(tmp, pathNum)
		*arr = append(*arr, tmp)
	}
	// 剪枝
	if len(pathNum)+n-start+1 < k {
		return
	}
	for i := start; i <= n; i++ {
		//	 判断以前是否收录过的代码，因为是先序遍历，所以同一层的数据true表示收录过
		pathNum = append(pathNum, i)
		combine_backtrack(pathNum, n, k, i+1, arr)
		//	 回溯
		pathNum = pathNum[:len(pathNum)-1]
	}
}

// 最小覆盖子串
func minWindow(s string, t string) string {
	var res string
	cnt := math.MaxInt32
	hashMap := make(map[byte]int)
	l := 0
	r := 0
	for i := 0; i < len(t); i++ {
		hashMap[t[i]]++
	}
	for r < len(s) {
		// 将s中出现的数据 在map里面-1
		hashMap[s[r]]--
		// 都要小于0才代表存在子串
		for check(hashMap) {
			// 判断当前的子串长度是否符合
			if r-l+1 < cnt {
				cnt = r - l + 1
				// 将新的子串赋值
				res = s[l : r+1]
			}
			// 当左指针需要离开的时候 +1
			hashMap[s[l]]++
			l++
		}
		r++
	}
	return res
}

func check(hashMap map[byte]int) bool {
	for _, v := range hashMap {
		if v > 0 {
			return false
		}
	}
	return true
}

// 颜色分类
func sortColors(arr []int) {
	QuickSort(arr, 0, len(arr)-1)
}

func partition(list []int, low, high int) int {
	p := list[low]
	for low < high {
		// 指针向左移动
		for low < high && p <= list[high] {
			high--
		}
		//填补low位置空值
		//high指针值 < pivot high值 移到low位置
		//high 位置值空
		list[low] = list[high]
		for low < high && p >= list[low] {
			low++
		}
		list[high] = list[low]
	}
	list[low] = p
	return low
}

func QuickSort(list []int, low, high int) {
	if low < high {
		p := partition(list, low, high)
		QuickSort(list, low, p-1)
		QuickSort(list, p+1, high)
	}
}

func searchMatrix(matrix [][]int, target int) bool {
	// 行
	len_row := len(matrix)
	// 列
	len_list := len(matrix[0])
	index := 0
	for index < len_row {
		// 在那一行的数据中
		if matrix[index][0] <= target && matrix[index][len_list-1] >= target {
			arr := matrix[index]
			search := sort.Search(len(arr), func(i int) bool {
				return arr[i] >= target
			})
			if arr[search] == target {
				return true
			}
			return false
		}
		index++
	}
	return false
}

// 矩阵置零
func setZeroes(matrix [][]int) {
	//	 将行列都设置一个bool值
	row_len := len(matrix)
	list_len := len(matrix[0])
	//	列
	list := make([]bool, list_len)
	// 	行
	row := make([]bool, row_len)
	for i := 0; i < row_len; i++ {
		for j := 0; j < list_len; j++ {
			if matrix[i][j] == 0 {
				// 标记那一行变为0
				list[j] = true
				row[i] = true
			}
		}
	}
	//	对行赋值
	index := 0
	for index < row_len {
		if row[index] == true {
			for i := 0; i < list_len; i++ {
				matrix[index][i] = 0
			}
		}
		index++
	}

	//	 对列赋值
	start := 0
	for start < list_len {
		if list[start] == true {
			for i := 0; i < row_len; i++ {
				matrix[i][start] = 0
			}
		}
		start++
	}
	fmt.Println(matrix)
}

// 编辑距离
func minDistance(word1 string, word2 string) int {

	// 因为插入 删除 替换的操作都是+1 的
	dp := make([][]int, len(word1)+1)
	for i := range dp {
		dp[i] = make([]int, len(word2)+1)
	}
	//	 初始化第一行 第一列
	for i := 0; i < len(word1)+1; i++ {
		dp[i][0] = i
	}
	for i := 0; i < len(word2)+1; i++ {
		dp[0][i] = i
	}
	for i := 1; i < len(word1)+1; i++ {
		for j := 1; j < len(word2)+1; j++ {
			// 判断上一个是否一样
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				//dp[i-1][j] 相当于在B的末尾添加一个元素
				// dp[i][j-1] 相当于在A的默认添加一个元素
				//dp[i-1][j]表示删除 dp[i][j-1]插入操作 dp[i-1][j-1]替换操作
				dp[i][j] = word_min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
			}
		}
	}
	return dp[len(word1)][len(word2)]
}
func word_min(c ...int) int {
	w_min := c[0]
	for _, item := range c {
		if item < w_min {
			w_min = item
		}
	}
	return w_min
}

func simplifyPath(path string) string {
	ret := []string{}
	for _, v := range strings.Split(path, "/") {
		switch v {
		case "":
			break
		case ".":
			break
		case "..":
			if l := len(ret); l > 0 {
				ret = ret[:l-1]
			}
		default:
			ret = append(ret, v)
		}
	}
	return "/" + strings.Join(ret, "/")
}

// 爬楼梯
func climbStairs(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 2
	}
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	dp[2] = 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]

}

// X的平方
func mySqrt(x int) int {
	// 从1开始进行输出
	end := 1
	// 先测试出来第一个超过这个的数
	for {
		sum := end * end
		if sum == x { // 等于直接返回
			return end
		} else if sum < x {
			// 为了快速查找 进行翻倍处理
			end = end * 2
			continue
		}
		break
	}
	m := 0
	start := end / 2
	for start <= end {
		mid := (start + end) / 2
		m = mid * mid
		if m == x {
			return mid
		} else if m < x {
			start = mid + 1
		} else {
			end = mid - 1
		}
	}
	return end
}

// 二进制求和
func addBinary(a string, b string) string {
	ai, _ := new(big.Int).SetString(a, 2)
	bi, _ := new(big.Int).SetString(b, 2)
	ai.Add(ai, bi)
	return ai.Text(2)
}

// 数据+1
func plusOne(digits []int) []int {
	for i := len(digits) - 1; i >= 0; i-- {
		// 计算当前的数据
		digits[i]++
		if digits[i]/10 == 0 {
			return digits
		}
		digits[i] = 0
	}
	// 初始数组的第一个数是9，并且后面一位会进一位
	return append([]int{1}, digits...)
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
