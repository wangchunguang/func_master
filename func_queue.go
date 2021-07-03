package func_master

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
