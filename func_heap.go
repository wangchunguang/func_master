package func_master

import "sort"

type Heap struct {
	data  []int
	count int
}

// NewHeap 初始化堆，第一个元素不用
func NewHeap() *Heap {
	// 第一个位置空着，不使用
	key := make([]int, 1)
	return &Heap{data: key, count: 0}
}

// Parent 父节点的位置
func (h *Heap) Parent(root int) int {
	return root / 2
}

// 左子树的位置
func (h *Heap) left(root int) int {
	return root * 2
}

// 右子树的位置
func (h *Heap) right(root int) int {
	return root*2 + 1
}

// Pop 返回堆顶元素
func (h *Heap) Pop() int {
	return h.data[1]
}

// 堆化的时候交换元素
func (h *Heap) exchange(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *Heap) Count() int {
	return h.count
}

// 大根堆操作

// PushMax 新增元素 大根堆
func (h *Heap) PushMax(v int) {
	h.count++
	h.data = append(h.data, v)
	// 元素新增之后进行堆化
	h.swimMax(h.count)
}

// PushListMax 在大根堆里面新增多个元素
func (h *Heap) PushListMax(arr []int) {
	for _, value := range arr {
		h.PushMax(value)
	}
}

// DelMax 删除大根堆的头部元素
func (h *Heap) DelMax() int {
	max := h.data[1]
	// 将最后一个元素放到第一个堆顶位置，然后向下进行堆化
	h.exchange(1, h.count)
	h.data = h.data[:h.count]
	h.count--
	h.sinkMax(1)
	return max
}

// 向上堆化 大根堆
func (h *Heap) swimMax(key int) {
	// 根节点<当前节点
	// 堆化
	for key > 1 && h.data[key] > h.data[h.Parent(key)] {
		h.exchange(h.Parent(key), key)
		key = h.Parent(key)
	}
}

//向下进行堆化
func (h *Heap) sinkMax(key int) {
	// 下沉到堆底
	for h.left(key) <= h.count {
		order := h.left(key)
		if h.right(key) <= h.count && h.data[order] < h.data[h.right(key)] {
			order = h.right(key)
		}
		// 节点比两个子节点都大,就不必下沉了
		if h.data[order] < h.data[key] {
			break
		}
		h.exchange(key, order)
		key = order
	}
}

// 小根堆操作

// PushMin 向小根堆插入元素
func (h *Heap) PushMin(num int) {
	h.count++
	h.data = append(h.data, num)
	h.swimMin(h.count)
}

// PushListMin 向小根堆插入多个元素
func (h *Heap) PushListMin(arr []int) {
	for _, value := range arr {
		h.PushMin(value)
	}
}

// DelMin 删除堆顶元素并返回
func (h *Heap) DelMin() int {
	min := h.data[1]
	h.exchange(1, h.count)
	h.data = h.data[:h.count]
	h.count--
	h.sinkMin(1)
	return min
}

// 从下往上进行堆化 新增元素的时候进行使用
func (h *Heap) swimMin(key int) {
	//	向上进行堆化，如果小于则交换位置 如果大于则直接跳过
	//	 当前值 小于父节点的值 才交换位置
	for key > 1 && h.data[key] < h.data[h.Parent(key)] {
		h.exchange(h.Parent(key), key)
		key = h.Parent(key)
	}
}

// 从上往下进行堆化 取出堆顶元素的时候使用
func (h *Heap) sinkMin(key int) {
	// 将堆顶的数据取出来之后，将最后一个数据放在堆的顶部，然后向下进行堆化
	//	因为堆的话算是完全二叉树，右子树不一定有 但是左子树一定有
	for h.left(key) <= h.count {
		order := h.left(key)
		// 查看右子树是否存在 在左子树和右子树中选择较小的哪一个
		if h.right(key) <= h.count && h.data[order] > h.data[h.right(key)] {
			order = h.right(key)
		}
		//	如果当前节点比选取出来的最小的节点都小，则直接跳过
		if h.data[order] > h.data[key] {
			break
		}
		//	 否则交换位置 继续进行向下堆化
		h.exchange(key, order)
		key = order
	}

}

// DeleteNum 删除堆中的其中一个指定大小的元素
func (h *Heap) DeleteNum(num int) {
	sort.Sort(sort.Reverse(sort.IntSlice(h.data[1:])))
	if num == h.data[1] {
		h.DelMax()
		return
	}
	left := 1
	right := len(h.data)
	for left < right {
		mid := left + (right-left)/2
		//	 如果相等
		if num > h.data[mid] {
			left = mid + 1
			continue
		}
		if num < h.data[mid] {
			right = mid - 1
			continue
		}
	}
}

// 单调队列
type MonotonousQueue struct {
	queue []int
}

func NewMonotonousQueue() *MonotonousQueue {
	return &MonotonousQueue{
		queue: make([]int, 0),
	}
}

func (m *MonotonousQueue) Front() int {
	return m.queue[0]
}

func (m *MonotonousQueue) Back() int {
	return m.queue[len(m.queue)-1]
}

func (m *MonotonousQueue) Empty() bool {
	return len(m.queue) == 0
}

func (m *MonotonousQueue) Push(val int) {
	// 当队列不为空，并且大于当前队列的最后一个元素时，循环执行下面的逻辑
	// Back表示获取当前队列的最后一个元素
	for !m.Empty() && val > m.Back() {
		m.queue = m.queue[:len(m.queue)-1]
	}
	m.queue = append(m.queue, val)
}

func (m *MonotonousQueue) Pop(val int) {
	// 如果当前传入的值等于队列的第一个元素，那么就将数组重新截取，
	// 如果不等于队列的头部 直接返回
	if !m.Empty() && val == m.Front() {
		m.queue = m.queue[1:]
	}
}
