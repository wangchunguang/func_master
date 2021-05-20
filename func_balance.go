package func_master

// 负载均衡的实现
// 实现算法：加权轮询负载均衡

type BalanceServer struct {
	//	 主机地址
	Host string
	//	 主机名
	Name string
	//	 权重
	Weight int
	// 	主机是否在线
	Onlice bool
}

type BalanceWeighted struct {
	Server *BalanceServer
	// 该服务器的权重，一般采用的是cpu数量
	Werght int
	// 服务器目前的权重，初始值为0
	CurrentWeight   int
	EffectiveWeight int
}

// LoadBalance 接口定义
type LoadBalance interface {
	//	 选择一个服务器，参数remove是需要排除选择的后端server
	Select(remove []string) *BalanceServer
	//	 更新可用的Server列表
	UpdateServers(servers []*BalanceServer)
}

type LoadBalanceWeightedRoundRobin struct {
	servers  map[string]*BalanceServer
	weighted map[string]*BalanceWeighted
}

// UpdateServers 更新
func (load *LoadBalanceWeightedRoundRobin) UpdateServers(servers map[string]*BalanceServer) {
	if len(load.servers) == len(servers) {
		for _, server := range servers {
			if old, ok := load.servers[server.Name]; ok { //当前服务存在etcd的服务中心
				if server.Host == old.Host && server.Weight == old.Weight && server.Onlice == old.Onlice {
					continue
				}
			}
			goto loadServer
		}
		return
	}

loadServer:
	weighted := make(map[string]*BalanceWeighted, 0)
	server := make(map[string]*BalanceServer, 0)
	for _, value := range servers {
		if value.Onlice == true {
			w := &BalanceWeighted{
				Server:          value,
				Werght:          value.Weight,
				CurrentWeight:   0,
				EffectiveWeight: value.Weight,
			}
			weighted[value.Name] = w
			server[value.Name] = value
		}

	}
	load.servers = server
	load.weighted = weighted
}

// Select 查询
func (load *LoadBalanceWeightedRoundRobin) Select(remove []string) *BalanceServer {
	if len(load.weighted) == 0 {
		return nil
	}

}

func (load *LoadBalanceWeightedRoundRobin) nextWeighted(weighted map[string]*BalanceWeighted, remove []string) (base *BalanceWeighted) {

}
