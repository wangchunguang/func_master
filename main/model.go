package main

type ListNode struct {
	Val  int
	Next *ListNode
}
type Item interface {
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func NewTreeNode() *TreeNode {
	return &TreeNode{}
}

func AddTreeNode() *TreeNode {
	return &TreeNode{5,
		//&TreeNode{9, &TreeNode{23, nil, nil}, &TreeNode{8, nil, nil}},
		&TreeNode{3, &TreeNode{2, nil, nil}, &TreeNode{4, nil, nil}},
		&TreeNode{6, nil, &TreeNode{7, nil, nil}}}
	//return &TreeNode{1,
	//	&TreeNode{2, nil, &TreeNode{3, nil,
	//		&TreeNode{4, nil, &TreeNode{5, nil, nil}}}},
	//	nil}

}

type Node struct {
	Val   int
	Prev  *Node
	Next  *Node
	Child *Node
}

func AddListNode() *ListNode {
	//head := &ListNode{1, &ListNode{2, &ListNode{3,&ListNode{4,&ListNode{5,nil}}}}}
	head := &ListNode{7, &ListNode{2, &ListNode{4,
		&ListNode{3, nil}}}}
	return head
}
func AddListNode1() *ListNode {
	//head := &ListNode{1, &ListNode{2, &ListNode{3,&ListNode{4,&ListNode{5,nil}}}}}
	head := &ListNode{5, &ListNode{6, &ListNode{4, nil}}}
	return head
}

func AddNode() *Node {

	node := &Node{1, &Node{2, &Node{6, &Node{3, &Node{4, nil, nil, nil}, nil, nil}, nil, nil}, nil, nil}, nil, nil}
	return node
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

func numbers(i, j int) int {
	if i > j {
		return i - j
	}
	return j - i
}

// SegmentTree 线段树
type SegmentTree struct {
	Tree   []int
	Arr    []int
	Lenght int
}

func NewSegmentTree(arr []int) *SegmentTree {
	tree := SetSegmentTree(arr)
	return &SegmentTree{Tree: tree, Lenght: len(arr)}
}

func SetSegmentTree(arr []int) []int {
	lenght := len(arr)
	tree := make([]int, lenght*2)
	// 先对线段树的后半段排序
	for i, j := lenght, 0; i < 2*lenght; i, j = i+1, j+1 {
		tree[i] = arr[j]
	}
	//	 所以左子树等于i*2 右子树等于i*2+1
	for i := lenght - 1; i > 0; i-- {
		tree[i] = tree[i*2] + tree[i*2+1]
	}
	return tree
}

func (this *SegmentTree) Update(index int, val int) {
	//	 先定位到指定的位置
	cur := this.Lenght + index
	this.Tree[cur] = val
	for cur > 0 {
		left := cur
		right := cur

		if cur%2 == 0 {
			// 如果当前是左子树
			right = cur + 1
		} else {
			//	 如果当前是右子树
			left = cur - 1
		}
		this.Tree[cur/2] = this.Tree[left] + this.Tree[right]
		cur = cur / 2
	}
}

func (this *SegmentTree) SunRange(left, right int) int {
	//	 先判断给定的位置在原始的位置上面的位置
	start := this.Lenght + left
	end := this.Lenght + right
	sum := 0
	for start <= end {
		if start%2 == 1 {
			//	 表示右子树
			sum += this.Tree[start]
			start++
		}
		if end%2 == 0 {
			sum += this.Tree[end]
			end--
		}
		start /= 2
		end /= 2
	}
	return sum
}

type NestedInteger struct {
	val  string
	next *NestedInteger
}
