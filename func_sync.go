package func_master

import "sync/atomic"

type WaitGroup struct {
	count int64
}

func (w *WaitGroup) Add(count int) {
	atomic.AddInt64(&w.count, int64(count))
}

func (w *WaitGroup) Done() {
	w.Add(-1)
}

func (w *WaitGroup) Wait() {
	if atomic.LoadInt64(&w.count) > 0 {
		Sleep(1)
	}
}

func (w *WaitGroup) TryWait() bool {
	return atomic.LoadInt64(&w.count) == 0
}
