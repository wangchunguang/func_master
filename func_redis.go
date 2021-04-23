package func_master

import (
	"github.com/go-redis/redis"
	"io"
	"net"
	"sync"
)

type RedisConfig struct {
	Addr     string // 地址
	Password string // 密码
	PoolSize int    // 连接池数量
}

type RedisManager struct {
	dbs      map[int]*Redis    // 连接redis的数量
	subMap   map[string]*Redis // 推送的地址
	channels []string          // 订阅发布的频道名称
	fun      func(channel, data string)
	lock     sync.RWMutex
}

type Redis struct {
	*redis.Client
	pubsub  *redis.PubSub // 发布订阅
	conf    *RedisConfig
	manager *RedisManager
}

var redisManagers []*RedisManager

// 添加redis连接的数量
func (r *RedisManager) Add(id int, config *RedisConfig) {
	r.lock.Lock()
	defer r.lock.Unlock()
	if _, ok := r.dbs[id]; ok {
		LogError("redis already exists id :=%v", id)
		return
	}
	re := &Redis{
		Client: redis.NewClient(&redis.Options{
			Addr:     config.Addr,
			Password: config.Password,
			PoolSize: config.PoolSize,
		}),
		conf:    config,
		manager: r,
	}
	re.WrapProcess(func(oldProcess func(cmd redis.Cmder) error) func(cmd redis.Cmder) error {
		return func(cmd redis.Cmder) error {
			err := oldProcess(cmd)
			if err != nil {
				_, retry := err.(net.Error)
				if !retry {
					retry = err == io.EOF
				}
				if retry {
					err = oldProcess(cmd)
				}
			}
			return err
		}
	})
	if _, ok := r.subMap[config.Addr]; !ok {
		r.subMap[config.Addr] = re
		if len(r.channels) > 0 {
			//	 订阅
			pubsub := re.Subscribe(r.channels...)
			// 在发布任何内容之前，请等待确认已创建订阅。
			_,err := pubsub.Receive()
			if err != nil {
				LogError("Please create a subscription %v",err)
				panic(err)
			}
			re.pubsub = pubsub
			goForRedis(func() {
				for IsRuning() {
					ch := pubsub.Channel()
					for msg := range ch{
						Go(func() {
							r.fun(msg.Channel, msg.Payload)
						})
					}
				}
			})
		}
	}
	r.dbs[id] = re
	LogInfo("connect to redis =%v", config.Addr)

}

// 初始化
func NewRedisManager(conf *RedisConfig) *RedisManager {
	redisManager := &RedisManager{
		dbs:    map[int]*Redis{},
		subMap: map[string]*Redis{},
	}
	redisManager.Add(0, conf)
	redisManagers = append(redisManagers, redisManager)
	return redisManager
}

// 检查id是否存在
func (r *RedisManager) CheckByRid(id int) bool {
	r.lock.RLock()
	defer r.lock.RUnlock()
	if _, ok := r.dbs[id]; !ok {
		return false
	}
	return true
}

// 获取db
func (r *RedisManager) GetByDB(id int) *Redis {
	r.lock.RLock()
	defer r.lock.RUnlock()
	db, ok := r.dbs[id]
	if !ok {
		panic(errIdMap)
	}
	return db
}

// 关闭所有redis
func (r *RedisManager) close() {
	for _, v := range r.dbs {
		if v.pubsub != nil {
			v.pubsub.Close()
		}
		v.Close()
	}
}

// 订阅
func (r *RedisManager) Sub(fun func(channel, data string), channels ...string) {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.channels = channels
	r.fun = fun
	for _, v := range r.subMap {
		// 将已经存在的订阅关闭，重新添加进新的channel
		if v.pubsub != nil {
			v.pubsub.Close()
		}
		pubsub := v.Subscribe(channels...)
		// 在发布任何内容之前，请等待确认已创建订阅。
		_,err := pubsub.Receive()
		if err != nil {
			LogError("Please create a subscription %v",err)
			panic(err)
		}
		v.pubsub = pubsub
		goForRedis(func() {
			for IsRuning() {
				ch := pubsub.Channel()
				for msg := range ch {
					Go(func() {
					fun(msg.Channel, msg.Payload)
					})

				}
			}
		})
	}
}

// 获取指定的channel上面有多少的订阅者
func (r *RedisManager) PubSubNumSub(channel ...string) ([]map[string]int64, error) {
	r.lock.Lock()
	defer r.lock.Unlock()
	var num []map[string]int64
	for _, v := range r.subMap {
		result, err := v.PubSubNumSub(channel...).Result()
		if err != nil {
			LogError("redis Failed to get data ")
			return nil, ErrDBDataType
		}
		num = append(num, result)
	}
	return num, nil
}
