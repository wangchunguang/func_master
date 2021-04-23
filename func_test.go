package func_master

import (
	"fmt"
	"os"
	"os/signal"
	"sync/atomic"
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
	var name int32
	swapInt32 := atomic.CompareAndSwapInt32(&name, 3, 5)
	fmt.Println(swapInt32)
	fmt.Println(name)
}

type Student struct {
	Name             string
}

type GetInterface interface {
	GetName() string
}

func (s *Student)GetName() string  {
	return ""
}

type S struct {
	Student
}

func (s *S)GetName() string    {

	return ""
}

