package func_master

// 队列实现栈
type MyStack struct {
	Stack []int
}

func Constructor() MyStack {
	return MyStack{[]int{}}
}

func (this *MyStack) Push(x int) {
	this.Stack = append(this.Stack, x)
}

func (this *MyStack) Pop() int {
	if this.Empty() {
		return 0
	}
	res := this.Stack[len(this.Stack)-1]
	this.Stack = this.Stack[:len(this.Stack)-1]
	return res
}

func (this *MyStack) Top() int {
	if this.Empty() {
		return 0
	}
	return this.Stack[len(this.Stack)-1]
}

func (this *MyStack) Empty() bool {
	if len(this.Stack) == 0 {
		return true
	}
	return false

}
