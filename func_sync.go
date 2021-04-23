package func_master

import "sync/atomic"

type WaitGroup struct {
	count int64
}

func (r *WaitGroup) Add(count int) {
	atomic.AddInt64(&r.count, int64(count))
}

func (r *WaitGroup) Done() {
	r.Add(-1)
}

func (r *WaitGroup) Wait() {
	if atomic.LoadInt64(&r.count) > 0 {
		Sleep(1)
	}
}

func (r *WaitGroup) TryWait() bool {
	return atomic.LoadInt64(&r.count) == 0
}
