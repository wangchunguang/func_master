package func_master

import (
	"runtime"
	"sync"
)

var DefLog *Log //日志

var gocount int32 //goroutine数量
var goid uint32
var waitAllForRedis sync.WaitGroup
var stop int32 //停止标志
var stopChanForLog = make(chan struct{})
var poolGoCount int32
var stopChanForGo = make(chan struct{})
var waitAll = WaitGroup{}

var StartTick int64
var NowTick int64
var Timestamp int64


var msgqueId uint32 //消息队列id
var msgqueMapSync sync.Mutex
var msgqueMap = map[uint32]IMsgQue{}

func init() {

	runtime.GOMAXPROCS(runtime.NumCPU())
	DefLog = NewLog(10000, &ConsoleLogger{true})
	DefLog.SetLevel(LogLevelInfo)
	timerTick()
}
