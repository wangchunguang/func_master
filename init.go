package func_master

import (
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"
)

var DefLog *Log //日志

type Statis struct {
	GoCount     int
	MsgqueCount int
	StartTime   time.Time
	LastPanic   int
	PanicCount  int32
}

var (
	gocount     int32 //goroutine数量
	goid        uint32
	stop        int32 //停止标志
	poolGoCount int32
	msgqueId    uint32     //消息队列id
	etcdTimeout uint32 = 5 // 出则etcd的超时时间

)

var (
	StartTick int64
	NowTick   int64
	Timestamp int64
	WeekStart int64 = 1514736000 //修正:不同时区不同
)

var (
	stopChanForLog = make(chan struct{})
	stopChanForSys = make(chan os.Signal, 1)
	stopChanForGo  = make(chan struct{})
)

var (
	waitAllForRedis sync.WaitGroup
	msgqueMapSync   sync.Mutex
	waitAll         = WaitGroup{}
	atexitMapSync   sync.Mutex
)

var (
	stopCheckMap = struct {
		sync.Mutex
		M map[uint64]string
	}{M: map[uint64]string{}}
	msgqueMap   = map[uint32]IMsgQue{}
	atexitMap   = map[uint32]func(){}
	statis      = &Statis{}
	someTimeout = 300 * time.Second // 长连接时间
)

func init() {
	rand.Seed(time.Now().Unix())
	runtime.GOMAXPROCS(runtime.NumCPU())
	DefLog = NewLog(10000, &ConsoleLogger{true})
	DefLog.SetLevel(LogLevelInfo)
	timerTick()
	WeekStart = DateToUnix("2018-01-01 00:00:00") //2018/1/1
}
