package func_master

import (
	"sync"
	"time"
)

// 雪花算法生成唯一id
const (
	workerBits  uint8 = 10
	numberBits  uint8 = 12
	workerMax   int64 = -1 ^ (-1 << workerBits)
	numberMax   int64 = -1 ^ (-1 << numberBits)
	timeShift   uint8 = workerBits + numberBits
	workerShift uint8 = numberBits
	startTime   int64 = 1525705533000 // 如果在程序跑了一段时间修改了epoch这个值 可能会导致生成相同的ID
)

type Worker struct {
	mu        sync.Mutex
	Timestamp int64 // 上一次记录的时间搓
	WorkerId  int64 // 该节点的id
	Number    int64 // 当前毫秒生成的序列号，从0开始叠加，1毫秒最多生成4096个id
}

// NewWorker 雪花算法
func NewWorker(workerId int64) *Worker {
	if workerId < 0 || workerId > workerMax {
		panic("Worker ID excess of quantity")
		return nil
	}
	//	 生成新的节点
	return &Worker{
		Timestamp: 0,
		WorkerId:  workerId,
		Number:    0}
}

func (w *Worker) GetId() int64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	now := time.Now().UnixNano() / 1e6
	if w.Timestamp == now {
		w.Number++
		if w.Number > numberMax {
			for now <= w.Timestamp {
				now = time.Now().UnixNano() / 1e6
			}
		}
	} else {
		w.Number = 0
		w.Timestamp = now
	}
	ID := int64((now-startTime)<<timeShift | (w.WorkerId << workerShift) | (w.Number))
	return ID
}

// Bsearch 二分查找算法
func Bsearch(arr []int, value int, left, right int) int {
	if left > right {
		return -1
	}
	mid := left + (right-left)/2
	if arr[mid] == value {
		return arr[mid]
	} else if arr[left] < value {
		return Bsearch(arr, value, mid+1, right)
	} else {
		return Bsearch(arr, value, left, mid-1)
	}

}
