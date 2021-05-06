package func_master

import (
	"sync"
	"sync/atomic"
)

var poolChan = make(chan func(), 1)
var once sync.Once

func Go(fn func()) {
	id := atomic.AddUint32(&goid, 1)
	c := atomic.AddInt32(&gocount, 1)
	if DefLog.Level() <= LogLevelDebug {
		debugStr := LogSimpleStack()
		LogDebug("goroutine start id:%d count:%d from:%s", id, c, debugStr)
	}
	select {
	case poolChan <- fn:
		atomic.AddInt32(&poolGoCount, 1)
	default:
		go func() {
			Try(fn)
		}()
	}
	once.Do(func() {
		go func() {
			RunGO()
		}()
	})
}

func Go2(fn func(cstop chan struct{})) {
	Go(func() {
		Try(func() { fn(stopChanForGo) })
	})
}

func RunGO() {
	for {
		select {
		case fn := <-poolChan:
			Try(fn)
			atomic.AddInt32(&poolGoCount, -1)
		default:
			Sleep(1)
		}
	}
}

func Try(fn func()) {
	waitAll.Add(1)
	defer func() {
		if err := recover(); err != nil {
			LogError("error catch = %v", err)
		}
	}()
	fn()
	waitAll.Done()
}

func GoForLog(fn func(cstop chan struct{})) bool {
	if IsStop() {
		return false
	}
	waitAll.Add(1)
	go func() {
		fn(stopChanForLog)
		waitAll.Done()
	}()
	return true
}

func goForRedis(fn func()) {
	waitAllForRedis.Add(1)
	var debugStr string
	id := atomic.AddUint32(&goid, 1)
	c := atomic.AddInt32(&gocount, 1)
	if DefLog.Level() <= LogLevelDebug {
		debugStr = LogSimpleStack()
		LogDebug("goroutine start id:%d count:%d from:%s", id, c, debugStr)
	}
	go func() {
		Try(fn)
		waitAllForRedis.Done()
		c = atomic.AddInt32(&gocount, -1)
		if DefLog.Level() <= LogLevelDebug {
			LogDebug("goroutine end id:%d count:%d from:%s", id, c, debugStr)
		}
	}()
}
