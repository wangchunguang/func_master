package func_master

import "strings"

// 负载均衡的实现
// 实现算法：加权轮询负载均衡

type BalanceServer struct {
	//	 主机地址
	Host string
	// 主机名称
	Name string
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
	curName string
	servers map[string]*BalanceServer
}

// NewLoadBalanceServerRoundRobin 初始化
func NewLoadBalanceServerRoundRobin(servers map[string]*BalanceServer) *LoadBalanceServerRoundRobin {
	load.UpdateServers(servers)
	return load
}

// UpdateServers 更新
func (load *LoadBalanceServerRoundRobin) UpdateServers(servers map[string]*BalanceServer) {
	server := make(map[string]*BalanceServer, 0)
	for key, value := range servers {
		split := strings.Split(key, "/")
		poolmap, ok := gateWayMap[split[0]]
		if !ok {
			poolmap = make(map[string]*HttpPool)
		}
		connect, b := ClientConnect(value.Host, "tcp")
		if !b {
			continue
		}
		poolmap[key] = NewHttpPool(key, connect)
		gateWayMap[split[0]] = poolmap

		s := &BalanceServer{
			Host:            value.Host,
			Name:            value.Name,
			Weight:          value.Weight,
			CurrentWeight:   0,
			EffectiveWeight: value.Weight,
		}
		server[value.Name] = s
	}
	load.servers = server
}

// Select 查询
func (load *LoadBalanceServerRoundRobin) Select() *BalanceServer {
	if len(load.servers) == 0 {
		return nil
	}
	s := load.nextServer(load.servers)
	if s == nil {
		return nil
	}
	return s
}

// 轮询获取服务
func (load *LoadBalanceServerRoundRobin) nextServer(servers map[string]*BalanceServer) (best *BalanceServer) {
	total := 0
	for key, value := range servers {
		//计算当前状态下所有节点的effectiveWeight之和totalWeight
		total += value.EffectiveWeight
		//	计算CurrentWeight
		value.CurrentWeight += value.EffectiveWeight
		// 寻找权重最大值
		if best == nil || best.CurrentWeight < value.CurrentWeight {
			best = value
			load.curName = key
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
