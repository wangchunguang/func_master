package func_master

import (
	"errors"
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

func NewWorker(workerId int64) (*Worker, error) {
	if workerId < 0 || workerId > workerMax {
		return nil, errors.New("Worker ID excess of quantity")
	}
	//	 生成新的节点
	return &Worker{
		Timestamp: 0,
		WorkerId:  workerId,
		Number:    0}, nil
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
