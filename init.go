package func_master

import (
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"
)

const (
	MACHINE_ID = int64(1)
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
	etcdTimeout uint32 = 5 // 连接etcd的超时时间
	maxCpuNum   int        // cpu数量
	onec        int
)

var (
	StartTick int64
	NowTick   int64
	Timestamp int64
	WeekStart int64  = 1514736000 //修正:不同时区不同
	HostName  string              // 服务名称前缀，同一个服务多台机器上面部署，所以通过服务前缀，去获取这个服务的所有ip端口，
)

var (
	StopChanForLog = make(chan struct{})
	StopChanForSys = make(chan os.Signal, 1)
	StopChanForGo  = make(chan struct{})
)

var (
	WaitAllForRedis sync.WaitGroup
	MsgqueMapSync   sync.Mutex
	WaitAll         = WaitGroup{}
	AtexitMapSync   sync.Mutex
	ServerMutex     sync.Mutex
)

var (
	msgqueMap  = map[int64]IMsgQue{}
	atexitMap  = map[uint32]func(){}
	cmdMap     = make(map[int]string)
	gateWayMap = make(map[string]map[string]*BalanceServer)
	newsMap    = make(map[int32]funcMessage)
)

var (
	stopCheckMap = struct {
		sync.Mutex
		M map[uint64]string
	}{M: map[uint64]string{}}
	statis      = &Statis{}
	someTimeout = 300 * time.Second // 长连接时间
	load        *LoadBalanceServerRoundRobin
	worker      *Worker
)

func init() {
	rand.Seed(time.Now().Unix())
	maxCpuNum = runtime.GOMAXPROCS(runtime.NumCPU())
	if maxCpuNum < 4 {
		maxCpuNum = 4
	}
	DefLog = NewLog(10000, &ConsoleLogger{true})
	DefLog.SetLevel(LogLevelInfo)
	timerTick()
	worker = NewWorker(MACHINE_ID)

	WeekStart = DateToUnix("2018-01-01 00:00:00") //2018/1/1

}
