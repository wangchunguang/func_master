package func_master

import "fmt"

// Node 链表节点
type Node struct {
	Value interface{}
	Next  *Node
}
type List struct {
	headNode *Node // 头结点
}

// IsEmpty 判断链表是否为空
func (list *List) IsEmpty() bool {
	if list.headNode == nil {
		return true
	}
	return false
}

// HeadSave 链表从头部插入元素
func (list *List) HeadSave(value interface{}) {
	node := &Node{Value: value}
	// 将就的头结点的指针指向新的节点里面的内存地址
	node.Next = list.headNode
	// 将新增的节点变为新的头结点
	list.headNode = node
}

// TailSave 从链表的尾部添加元素
func (list *List) TailSave(value interface{}) {
	// 先将节点赋值
	node := &Node{Value: value}
	// 链表为空，插入的尾节点为头部节点
	if list.IsEmpty() {
		list.headNode = node
		return
	}
	cur := list.headNode  // 定义变量存储头结点
	for cur.Next != nil { // 判断是否为尾节点，如果为nil表示为尾节点，
		cur = cur.Next // 每次将获取出来的节点赋值，当跳出循环的时候，cur存储的是最后一个节点
	}
	cur.Next = node // 将旧的尾节点的指针赋值新插入的尾节点地址
}

// Length 获取链表的长度
func (list *List) Length() int {
	count := 0
	cur := list.headNode
	for cur != nil {
		count++
		cur = cur.Next
	}
	return count
}

// Insert 在链表指定位置之后添加元素
func (list *List) Insert(index int, value interface{}) {
	cur := list.headNode
	if index <= 0 { // 表示在头部插入节点
		list.HeadSave(value)
		return
	} else if index > list.Length() { // 表示在尾部添加数据
		list.TailSave(value)
		return
	}
	count := 0
	// -1 是因为插入节点要注意顺序，先将插入的节点的指针指向插入节点的后一位指针，然后再讲插入节点的前一位的指针赋值插入数据的内存地址，不然会造成指针丢失
	for count < (index - 1) {
		count++
		cur = cur.Next // 获取需要插入节点的前一个节点
	}
	// 获取插入节点位置后的节点
	pur := cur.Next
	node := &Node{Value: value}
	node.Next = pur // 先将需要插入节点的后一个节点指针指向插入的节点
	cur.Next = node // 将原来位置节点的下一个指针指向需要插入节点的地址
}

// DeleteAtIndex 删除指定位置的节点的元素
func (list *List) DeleteAtIndex(index int) {
	// 将头结点提取出来
	cur := list.headNode
	if index <= 0 {
		cur = cur.Next
	} else if index > list.Length() { // 如果删除的位置大于链表的长度就直接返回
		return
	}
	count := 0
	for count < (index - 1) {
		cur = cur.Next //获取删除节点的位置的元素
		count++
	}
	//	 获取删除节点下一个节点的指针
	cur.Next = cur.Next.Next
}

// RangeList 遍历节点中的所有元素
func (list *List) RangeList() {
	if list.IsEmpty() {
		fmt.Println(nil)
		return
	}
	cur := list.headNode
	for {
		fmt.Println(cur.Value)
		if cur.Next != nil {
			cur = cur.Next
		} else {
			break
		}
	}
}

// DeleteAtValue 删除指定的值
func (list *List) DeleteAtValue(value interface{}) {
	// 将头结点提取出来
	cur := list.headNode
	if cur.Value == value {
		cur = cur.Next.Next
	} else {
		// 循环遍历
		for cur.Next != nil {
			if cur.Next.Value == value {
				cur.Next = cur.Next.Next
			} else {
				cur = cur.Next
			}
		}
	}
}

// ReverseList 链表的反转 可以尝试从当前的节点的头结点去头部新增元素，链表就形成了以初始的头结点像两边扩展
func (list *List) ReverseList() *List {
	if list.IsEmpty() {
		return nil
	}
	head := &List{}
	cur := list.headNode
	for cur != nil {
		head.HeadSave(cur.Value)
		cur = cur.Next
	}
	return head
}
