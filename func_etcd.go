package func_master

import (
	"context"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"go.etcd.io/etcd/clientv3"

	"sync"
	"time"
)

// ServiceRegister 主要实现服务注册，健康检查，服务发现
type ServiceRegister struct {
	cli     *clientv3.Client //etcd client
	leaseId clientv3.LeaseID // 租约id
	//	 租约keepAlive相应的chan
	keepAliveChan <-chan *clientv3.LeaseKeepAliveResponse
	key           string // key
	val           string // value
}

// NewServiceRegister 新建注册服务 endpoints 端口列表
func NewServiceRegister(endpoints []string, key, val string, lease int64) (*ServiceRegister, error) {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: time.Duration(etcdTimeout) * time.Second,
	})
	if err != nil {
		LogError("etcd registration failed endpoints :%v err :%s", endpoints, err)
		return nil, err
	}
	ser := &ServiceRegister{
		cli: client,
		key: key,
		val: val,
	}
	// 申请租约设置时间keepalive
	if err := ser.putKeyWithLease(lease); err != nil {
		return nil, err
	}
	return ser, err
}

// 设置租约
func (s *ServiceRegister) putKeyWithLease(lease int64) error {
	// 创建一个新的租约
	grant, err := s.cli.Grant(context.Background(), lease)
	if err != nil {
		LogError("Failed to create %s", err)
		return err
	}
	//	注册服务并设置租约  同时将租约id放在注册服务后面
	_, err = s.cli.Put(context.Background(), s.key, s.val, clientv3.WithLease(grant.ID))
	if err != nil {
		LogError("Failed to register service err :%s ", err)
		return err
	}
	// 设置续约，定时发送需求请求  如果发布到该通道的keepalive响应没有立即被使用，则租约客户端将至少每秒钟继续向etcd服务器发送一个保持活动的请求
	alive, err := s.cli.KeepAlive(context.Background(), grant.ID)
	if err != nil {
		LogError("Lease renewal failed err %s", err)
		return err
	}
	s.leaseId = grant.ID
	s.keepAliveChan = alive
	return nil
}

// ListenLeaseRespChan 监听 续约情况
func (s *ServiceRegister) ListenLeaseRespChan() {
	for leaseKeepResp := range s.keepAliveChan {
		LogInfo("Successful renewal %s", leaseKeepResp)
	}
	LogInfo("Close renewal")
}

// Close 注销服务
func (s *ServiceRegister) Close() error {
	// 先撤销租约
	if _, err := s.cli.Revoke(context.Background(), s.leaseId); err != nil {
		LogError("Failed to revoke the lease err :%s", err)
		return err
	}
	// 关闭etcd连接
	return s.cli.Close()
}

// ServiceDiscovery 服务发现
type ServiceDiscovery struct {
	cli        *clientv3.Client // etcd client
	serverList sync.Map         // 服务列表
}

// NewServiceDiscovery 新建发现服务
func NewServiceDiscovery(endpoints []string) (*ServiceDiscovery, error) {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: time.Duration(etcdTimeout) * time.Second,
	})
	if err != nil {
		LogError("Service discovery failed err :%s", err)
		return nil, err
	}

	ser := &ServiceDiscovery{
		cli: client,
	}
	return ser, nil
}

// WatchService 监视和初始化服务器列表
func (sd *ServiceDiscovery) WatchService(prefix string) error {
	//	根据前缀获取现有的key
	response, err := sd.cli.Get(context.Background(), prefix, clientv3.WithPrefix())
	if err != nil {
		LogError("Get data error err :%s", err)
		return err
	}
	for _, kv := range response.Kvs {
		sd.serverList.Store(kv.Key, kv.Value)
	}
	//	监听前缀，表示是否随时进行修改
	go sd.watcher(prefix)
	return nil

}

// 监听前缀
func (sd *ServiceDiscovery) watcher(prefix string) {
	// 发布监听请求，等待新的通道
	watch := sd.cli.Watch(context.Background(), prefix, clientv3.WithPrefix())
	for wresp := range watch {
		for _, resp := range wresp.Events {
			// 返回的数据为两种，put表示新增进去，delete表示删除
			switch resp.Type {
			case mvccpb.PUT: // 修改或者新增
				LogInfo("put key =%s", resp.Kv.Key)
				sd.setServiceList(string(resp.Kv.Key), string(resp.Kv.Value))
			case mvccpb.DELETE: // 删除操作
				LogInfo("Delete key =%s", resp.Kv.Key)
				sd.delServiceList(string(resp.Kv.Key))
			}
		}
	}
}

// 新增服务地址
func (sd *ServiceDiscovery) setServiceList(key, value string) {
	sd.serverList.Store(key, value)
}

// 删除服务操作
func (sd *ServiceDiscovery) delServiceList(key string) {
	sd.serverList.Delete(key)
}

// 读取操作单个服务器地址
func (sd *ServiceDiscovery) loadServiceList(key string) string {
	value, _ := sd.serverList.Load(key)
	return value.(string)
}

// 读取所有服务器地址
func (sd *ServiceDiscovery) loadListServiceList() []string {
	arr := make([]string, 0)
	f := func(key, value interface{}) bool {
		arr = append(arr, value.(string))
		return true
	}
	sd.serverList.Range(f)
	return arr
}

// 关闭服务
func (sd *ServiceDiscovery) Close() error {
	return sd.cli.Close()
}
