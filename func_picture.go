package func_master

import "fmt"

// 图
/**
 图分为有向图和无向图，可以理解树的节点为图的顶点，图中一个顶点到另一个顶点连接的叫做边，顶点和边连接的条数叫做顶点的度
图是一种非线性的数据结构比数更加复杂
在有向图中，度分为出度和入度，
*/

// Item 数据类型
type Item interface {
}

// PictureNode 组成图的顶点
type PictureNode struct {
	Value    Item
	Searched bool
}

func (p *PictureNode) String() string {
	return fmt.Sprintf("%v", p.Value)
}

// ItemGraph 定义图的结构 图由顶点与边组成P E
type ItemGraph struct {
	Nodes []*PictureNode                  // 顶点
	Edges map[*PictureNode][]*PictureNode // 边 一个顶点多个边
}

// AddNode 添加节点
func (item *ItemGraph) AddNode(node *PictureNode) {
	item.Nodes = append(item.Nodes, node)
}

// AddEdges 添加边
func (item *ItemGraph) AddEdges(u, v *PictureNode) {
	if item.Edges == nil {
		item.Edges = make(map[*PictureNode][]*PictureNode)
	}
	//	 无向图
	item.Edges[u] = append(item.Edges[u], v) // 表示u->v的边
	item.Edges[v] = append(item.Edges[v], u) // 表示v->u的边

}

// 输出图
func (item *ItemGraph) String() {
	s := ""
	for i := 0; i < len(item.Nodes); i++ {
		s += item.Nodes[i].String() + ":"
		near := item.Edges[item.Nodes[i]]

		for j := 0; j < len(near); j++ {
			s += near[j].String() + "->"
		}
		s += "\n"
	}
	fmt.Println(s)
}

// BFS 图的广度优先算法
func (item *ItemGraph) BFS(node *PictureNode) {
	// 存储待搜索的节点
	var adNodes []*PictureNode
	// 为true的时候表示已经被访问
	node.Searched = true
	fmt.Printf("%d:", node.Value)
	for _, i := range item.Edges[node] {
		// 获取图中指向的顶点是否被访问，被访问直接跳过，没有被访问加入带搜索的节点，形成入队，出队操作
		if !i.Searched {
			// 先设置为true 表示已经被访问
			i.Searched = true
			adNodes = append(adNodes, i)
			fmt.Printf("%v ", i.Value)

		}
	}
	fmt.Printf("\n")
	for _, i := range adNodes {
		item.BFS(i)
	}
}

// DFS 深度优先遍历 FDS
func (item *ItemGraph) DFS() {
	for _, node := range item.Nodes {
		if !node.Searched {
			node.Searched = true
			fmt.Printf("%v ->", node.Value)
			item.visitNode(node)
			fmt.Printf("\n")
			item.DFS()
		}
	}

}

// 获取单个节点的最深度
func (item *ItemGraph) visitNode(node *PictureNode) {
	for _, i := range item.Edges[node] {
		if !i.Searched {
			i.Searched = true
			fmt.Printf("%v->", i.Value)
			item.visitNode(i)
			return
		}
	}

}
