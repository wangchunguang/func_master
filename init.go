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
	etcdTimeout uint32 = 5 // 连接etcd的超时时间
	maxCpuNum   int        // cpu数量

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
	stopCheckMap = struct {
		sync.Mutex
		M map[uint64]string
	}{M: map[uint64]string{}}
	msgqueMap   = map[uint32]IMsgQue{}
	atexitMap   = map[uint32]func(){}
	statis      = &Statis{}
	someTimeout = 300 * time.Second // 长连接时间
	load        *LoadBalanceServerRoundRobin
	cmdMap      = make(map[int]string)
	gateWayMap  = make(map[string]map[string]*BalanceServer)
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
	WeekStart = DateToUnix("2018-01-01 00:00:00") //2018/1/1

}
