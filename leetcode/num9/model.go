package num9

type ListNode struct {
	Val  int
	Next *ListNode
}
type Item interface {
}

type TreeNode struct {
	Count int // 左边的值
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func numbers(i, j int) int {
	if i > j {
		return i - j
	}
	return j - i
}

func InsertNode(node, newNode *TreeNode, small_count *int) {
	if newNode.Val <= node.Val {
		node.Count++
		//	 左边插入
		if node.Left != nil {
			InsertNode(node.Left, newNode, small_count)
		} else {
			node.Left = newNode
		}
	}
	if newNode.Val > node.Val {
		// 统计左边的元素，+更节点的个数，再加一个
		*small_count = *small_count + node.Count + 1
		if node.Right != nil {
			InsertNode(node.Right, newNode, small_count)
		} else {
			node.Right = newNode
		}
	}
}

func NewTreeNode() *TreeNode {
	return &TreeNode{}
}

type Node struct {
	Val   int
	Prev  *Node
	Next  *Node
	Child *Node
}

func AddListNode() *ListNode {
	//head := &ListNode{1, &ListNode{2, &ListNode{3,&ListNode{4,&ListNode{5,nil}}}}}
	head := &ListNode{1, &ListNode{2, &ListNode{3,
		&ListNode{4, &ListNode{5, &ListNode{6, &ListNode{7, &ListNode{8, nil}}}}}}}}
	return head
}
func AddListNode1() *ListNode {
	//head := &ListNode{1, &ListNode{2, &ListNode{3,&ListNode{4,&ListNode{5,nil}}}}}
	head := &ListNode{3, &ListNode{2, &ListNode{4, nil}}}
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
