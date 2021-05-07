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
	c := make(chan int)
	select {
	case <-c:
	}
	fmt.Println(222222222)
}

func demo(num chan int) {

	num <- 12
	//c := make(chan int, 1)
	//cstop := make(chan struct{})
	//select {
	//case n, ok := <-cstop:
	//	fmt.Println(n, ok)
	//case num := <-c:
	//	fmt.Println(num)
	//}

}
