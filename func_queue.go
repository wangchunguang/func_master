package func_master

import "fmt"

// ArrayQueue 队列
type ArrayQueue struct {
	// 采用队列实现
	Queue []interface{}
	// 队列的长度
	Count int
	// 对头下标
	Head int
	// 队尾下标
	Tail int
}

// NewArrayQueue 初始化
func NewArrayQueue() *ArrayQueue {
	return &ArrayQueue{
		Queue: []interface{}{},
		Count: 0,
		Head:  0,
		Tail:  0,
	}
}

// Insert 添加到队列
func (queue *ArrayQueue) Insert(value interface{}) {
	queue.Queue = append(queue.Queue, value)
	queue.Count = len(queue.Queue)
	queue.Tail++
}

// DeQueue 出队
func (queue *ArrayQueue) DeQueue() interface{} {
	if queue.Head == queue.Tail {
		return nil
	}
	// 不采用切片拷贝，那样消耗性能
	value := queue.Queue[queue.Head]
	queue.Head++
	queue.Count = len(queue.Queue)
	return value
}

// 用栈实现队列

type MyQueue struct {
	//	 准备两个栈，将数据放入栈中，压栈执行将数据放入另一个栈中
	stack_in  []int
	stack_out []int
}

func Constructor_Queue() MyQueue {
	return MyQueue{[]int{}, []int{}}
}

func (this *MyQueue) Push(x int) {
	this.stack_in = append(this.stack_in, x)
}

func (this *MyQueue) inToOut() {
	fmt.Println(this.stack_in)
	for len(this.stack_in) != 0 {
		this.stack_out = append(this.stack_out, this.stack_in[len(this.stack_in)-1])
		this.stack_in = this.stack_in[:len(this.stack_in)-1]
	}
	fmt.Println(this.stack_out)
}

func (this *MyQueue) Pop() int {
	if len(this.stack_out) == 0 {
		this.inToOut()
	}
	if len(this.stack_out) == 0 {
		return 0
	}
	num := this.stack_out[len(this.stack_out)-1]
	this.stack_out = this.stack_out[:len(this.stack_out)-1]
	return num

}

func (this *MyQueue) Peek() int {
	if len(this.stack_out) == 0 {
		this.inToOut()
	}
	if len(this.stack_out) == 0 {
		return 0
	}
	return this.stack_out[len(this.stack_out)-1]
}

func (this *MyQueue) Empty() bool {
	if len(this.stack_out) == 0 && len(this.stack_in) == 0 {
		return true
	}
	return false
}
