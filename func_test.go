package func_master

import (
	"encoding/binary"
	"fmt"
	"os"
	"os/signal"
	"reflect"
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
	signal.Notify(stopChanForSys, os.Interrupt, os.Kill, syscall.SIGTERM)
	fmt.Println(11111111111)

	select {
	case s := <-stopChanForSys:
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
	tick := time.NewTimer(time.Second * 5)
	select {
	case <-stopChanForGo:
		fmt.Println(111111111)
	case n := <-c:
		fmt.Println(n)
	case <-tick.C:
		fmt.Println(8888888888)

	}
	fmt.Println(20000000)

}

func demo(num chan int) {
	time.Sleep(5 * time.Second)
	close(num)
}

func TestLogError(t *testing.T) {
	ch := make(chan int)
	go demo(ch)
	for i := range ch {
		fmt.Println(i)
	}
	fmt.Println(222222)
}

func TestEtcd(t *testing.T) {
	var endpoints = []string{"localhost:2379"}
	ser, err := NewServiceRegister(endpoints, "/web/nodel", "localhost:8000", 5)
	if err != nil {
		LogError(err)
		return
	}
	go ser.ListenLeaseRespChan()

	select {
	// case <-time.After(20 * time.Second):
	// 	ser.Close()
	}

}
