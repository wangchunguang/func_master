package func_master

import (
	"net"
	"strings"
)

// BalanceServer 负载均衡的实现
// 实现算法：加权轮询负载均衡
type BalanceServer struct {
	//	 主机地址
	Host string
	// 主机名称
	Name string
	// 读取服务
	Coon net.Conn
	// 该服务器的权重，一般采用的是cpu数量
	Weight int
	// 服务器目前的权重，初始值为0
	CurrentWeight int
	// 有效权重，初始值为weight, 通讯过程中发现节点异常，则-1 ，之后再次选取本节点，调用成功一次则+1，直达恢复到weight 。 用于健康检查，处理异常节点，降低其权重。
	EffectiveWeight int
}

// LoadBalance 接口定义
type LoadBalance interface {
	//	 选择一个服务器，参数remove是需要排除选择的后端server
	Select() *BalanceServer
	//	 更新可用的Server列表
	UpdateServers(servers map[string]*BalanceServer)
}

type LoadBalanceServerRoundRobin struct {
	servers map[string]map[string]*BalanceServer
}

// NewLoadBalanceServerRoundRobin 初始化
func NewLoadBalanceServerRoundRobin(servers map[string]*BalanceServer) *LoadBalanceServerRoundRobin {
	load.UpdateServers(servers)
	return load
}

// UpdateServers 更新
func (load *LoadBalanceServerRoundRobin) UpdateServers(servers map[string]*BalanceServer) {
	balanceMap := map[string]*BalanceServer{}
	for key, value := range servers {
		// 每个服务都会有多台服务器，例如 route/1  route/2 ...
		split := strings.Split(key, "/")
		// gateWayMap里面 每一个服务发布了多个服务，
		balanceMap, _ = gateWayMap[split[0]]
		if len(balanceMap) == 0 {
			balanceMap = make(map[string]*BalanceServer)
		}
		conn, b := ClientConnect(value.Host, "tcp")
		if !b {
			continue
		}
		s := &BalanceServer{
			Host:            value.Host,
			Name:            value.Name,
			Coon:            conn,
			Weight:          value.Weight,
			CurrentWeight:   0,
			EffectiveWeight: value.Weight,
		}
		balanceMap[key] = s
		gateWayMap[split[0]] = balanceMap
	}
	load.servers = gateWayMap
}

// Select 查询
func (load *LoadBalanceServerRoundRobin) Select(addr string) *BalanceServer {
	if len(load.servers) == 0 {
		return nil
	}
	if value, ok := load.servers[addr]; ok {
		s := load.nextServer(value)
		if s == nil {
			return nil
		}
		return s
	}
	return nil
}

// 轮询获取服务
func (load *LoadBalanceServerRoundRobin) nextServer(servers map[string]*BalanceServer) (best *BalanceServer) {
	total := 0
	ServerMutex.Lock()
	defer ServerMutex.Unlock()
	for _, value := range servers {
		//计算当前状态下所有节点的effectiveWeight之和totalWeight
		total += value.EffectiveWeight
		//	计算CurrentWeight
		value.CurrentWeight += value.EffectiveWeight
		// 寻找权重最大值
		if best == nil || best.CurrentWeight < value.CurrentWeight {
			best = value
		}
	}
	if best == nil {
		return nil
	}
	best.CurrentWeight -= total
	return best
}

func (load *LoadBalanceServerRoundRobin) ToString() {
	for key, value := range load.servers {
		LogInfo("%s = %+v", key, value)
	}
}
