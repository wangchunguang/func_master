package func_master

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"os/signal"
	"reflect"
	"strconv"
	"syscall"
	"testing"
	"time"
)

func Test_Func_Go(t *testing.T) {
	i := time.Now().UnixNano()
	for i := 0; i < 11000; i++ {
		Go(func() {
		})
	}
	//Go(func() {
	//	for i := 0; i < 10000; i++ {
	//		Sleep(1)
	//		fmt.Println()
	//	}
	//})
	//Go2(func(cstop chan struct{}) {
	//	for i := 0; i < 10000; i++ {
	//		Sleep(1)
	//		fmt.Println()
	//	}
	//})
	k := time.Now().UnixNano()
	fmt.Println(k - i)
	sg := make(chan os.Signal)
	signal.Notify(sg)
	s := <-sg
	fmt.Println(s)
}

func TestRedisManager_Add(t *testing.T) {
	r := &RedisConfig{
		Addr:     "127.0.0.1:6379",
		Password: "",
		PoolSize: 6,
	}
	manager := NewRedisManager(r)
	redis := manager.dbs[0]
	defer redis.Close()
	manager.Sub(recvPublish, "newdb")
	result, err := redis.Publish("newdb", "hello").Result()

	if err != nil {
		panic(err)
	} else {
		LogInfo(result)
	}
	sg := make(chan os.Signal)
	signal.Notify(sg)
	_ = <-sg

}

func recvPublish(channel string, data string) {
	if channel == "newdb" {
		fmt.Println("订阅成功")
	} else {
		fmt.Println("订阅失败")
	}
}

func TestTimeToStamp(t *testing.T) {
	res(&CmdTaskMsgTestC2S{})
}

func res(v interface{}) {
	msgType := reflect.TypeOf(v)
	fmt.Println(msgType)
	fmt.Println(msgType.Kind())
	fmt.Println(reflect.Ptr)
}

type CmdTaskMsgTestC2S struct {
	Test  string `match:"k"`
	Task  string `match:"k"`
	Gamer int
}

func TestWaitForSystemExit(t *testing.T) {
	signal.Notify(StopChanForSys, os.Interrupt, os.Kill, syscall.SIGTERM)
	fmt.Println(11111111111)

	select {
	case s := <-StopChanForSys:
		fmt.Println(s)
		fmt.Println(2222222222)

	}
	fmt.Println(6666666666)
}

func TestDaemon(t *testing.T) {
	//fmt.Println(reflect.TypeOf(CmdTaskMsgTestC2S{}))
	//num := 123
	cmd := CmdTaskMsgTestC2S{
		Test:  "123",
		Task:  "231",
		Gamer: 0,
	}
	valueOf := reflect.ValueOf(&cmd)
	fmt.Println(valueOf.Kind())
	switch valueOf.Kind() {
	case reflect.Ptr:
		fmt.Println(valueOf.IsNil())
		fmt.Println(valueOf.Elem())
	case reflect.Int:

	}
	fmt.Println(reflect.Int)
	fmt.Println(valueOf.Type())
	fmt.Println(valueOf.Kind())

}

func TestSetTimeout(t *testing.T) {
	StartTick = time.Now().UnixNano() / 1000000
	NowTick = StartTick
	Timestamp = NowTick / 1000
	u := uint32(Timestamp)
	u1 := uint32(Timestamp) + uint32(RandNumber(99999))

	data := make([]byte, 8)
	binary.BigEndian.PutUint32(data, u)
	fmt.Println(data)
	binary.BigEndian.PutUint32(data[4:], u1)
	fmt.Println(Timestamp)
	fmt.Println(u)
	fmt.Println(u1)
	fmt.Println(data)

}

func TestGetStatis(t *testing.T) {

	var num uint8
	num |= FlagEncrypt
	fmt.Println(num)
}

func TestWalkDir(t *testing.T) {
	tick := time.NewTimer(time.Second * time.Duration(1))
	fmt.Println("当前时间为111111111111:", time.Now())
	c := <-tick.C
	fmt.Println("当前时间为22222222222:", c)

}

func TestWaitForSystemExit2(t *testing.T) {
	c := make(chan int)
	go demo(c)
	for {
		select {
		case <-StopChanForGo:
			fmt.Println(111111111)
		case n := <-c:
			fmt.Println(n)

		default:

		}
	}
	select {}

}

func demo(num chan int) {
	for {
		num <- 5

	}
}

func TestLogError(t *testing.T) {
	ch := make(chan int)
	go demo(ch)
	for i := range ch {
		fmt.Println(i)
	}
	fmt.Println(222222)
}

//服务注册
func TestEtcd_server(t *testing.T) {
	var endpoints = []string{"localhost:2379"}
	web_key := "/web/nodel"
	value := "localhost:800"
	go func() {
		for i := 0; i < 10; i++ {
			select {
			case <-time.Tick(3 * time.Second):
				demo_etcd_derver(endpoints, web_key+"/"+strconv.Itoa(i), value+strconv.Itoa(i))
			}

		}
	}()
	grpc_key := "/grpc/nodel"
	go func() {
		for i := 0; i < 5; i++ {
			select {
			case <-time.Tick(2 * time.Second):
				demo_etcd_derver(endpoints, grpc_key+"/"+strconv.Itoa(i), value+strconv.Itoa(i))

			}
		}
	}()

	select {
	// case <-time.After(20 * time.Second):
	// 	ser.Close()
	}
}

func demo_etcd_derver(endpoints []string, key, value string) {
	_, err := NewServiceRegister(endpoints, key, value, 5)
	if err != nil {
		LogError(err)
		return
	}

}

//服务发现
func TestEtcd_service(t *testing.T) {
	var endpoints = []string{"localhost:2379"}
	ser, err := NewServiceDiscovery(endpoints)
	if err != nil {
		return
	}
	defer ser.Close()
	ser.WatchService("/")
	for {
		select {
		case <-time.Tick(2 * time.Second):
			log.Println(ser.loadListServiceList())
		}
	}
}

func TestGoTo(t *testing.T) {
	for i := 0; i < 10; i++ {
		fmt.Println(i)
		if i == 3 {
			goto build
		}
	}

build:
	fmt.Println(11111)
}

func TestLoadBalanceWeightedRoundRobin_Select(t *testing.T) {
	servers := make(map[string]*BalanceServer, 0)
	servers["0"] = &BalanceServer{Host: "192.186.0.1:8080", Name: "0", Weight: 4}
	servers["1"] = &BalanceServer{Host: "192.186.0.1:8081", Name: "1", Weight: 2}
	servers["2"] = &BalanceServer{Host: "192.186.0.1:8090", Name: "2", Weight: 1}
	//servers["3"] = &BalanceServer{Host: "192.186.0.1:9000",Name: "3",Weight: 8,Onlice: true}
	//servers["4"] = &BalanceServer{Host: "192.186.0.1:3000",Name: "4",Weight: 10,Onlice: true}
	//servers["5"] = &BalanceServer{Host: "192.186.0.1:4000",Name: "5",Weight: 10,Onlice: true}
	//servers["6"] = &BalanceServer{Host: "192.186.0.1:5000",Name: "6",Weight: 2,Onlice: true}
	//servers["7"] = &BalanceServer{Host: "192.186.0.1:6000",Name: "7",Weight: 12,Onlice: true}
	//servers["8"] = &BalanceServer{Host: "192.186.0.1:7000",Name: "8",Weight: 4,Onlice: true}

	lb := NewLoadBalanceServerRoundRobin(servers)
	for i := 0; i < 10; i++ {
		_ = lb.Select()
		//fmt.Println(s)
		lb.ToString()
	}

	Sleep(10000)

}

type teacher struct {
}

func (t *teacher) Num(bun float32) {

}

func (t *teacher) Set(bun float64) {

}
func (t *teacher) Get(i int) {

}

func TestStrEqualFold(t *testing.T) {
	te := &teacher{}
	demo1(te)

}

func demo1(v interface{}) {
	of := reflect.TypeOf(v)
	fmt.Println(of.NumMethod())
	for i := 0; i < of.NumMethod(); i++ {
		fmt.Println(of.Method(i).Name)
		fmt.Println(of.Method(i).Type)
	}
}
