package func_master

import "fmt"

// 图
/**
 图分为有向图和无向图，可以理解树的节点为图的顶点，图中一个顶点到另一个顶点连接的叫做边，顶点和边连接的条数叫做顶点的度
图是一种非线性的数据结构比数更加复杂
在有向图中，度分为出度和入度，
*/

type PictureNodeInterface interface {
	String()
	AddNode(node *PictureNode)
	AddEdges(u, v *PictureNode)
}

// Item 数据类型
type Item interface {
}

// PictureNode 组成图的顶点
type PictureNode struct {
	Value Item
}

func (p *PictureNode) String() string {
	return fmt.Sprintf("%v", p.Value)
}

// ItemGraph 定义图的结构 图由顶点与边组成P E
type ItemGraph struct {
	Nodes []*PictureNode                 // 顶点
	Edges map[PictureNode][]*PictureNode // 边 一个顶点多个边
}

// AddNode 添加节点
func (item *ItemGraph) AddNode(node *PictureNode) {
	item.Nodes = append(item.Nodes, node)
}

// AddEdges 添加边
func (item *ItemGraph) AddEdges(u, v *PictureNode) {
	if item.Edges == nil {
		item.Edges = make(map[PictureNode][]*PictureNode)
	}

	//	 无向图
	item.Edges[*u] = append(item.Edges[*u], v) // 表示u->v的边
	item.Edges[*v] = append(item.Edges[*v], u) // 表示v->u的边
}

// 输出图
func (item *ItemGraph) String() {
	s := ""
	for i := 0; i < len(item.Nodes); i++ {
		s += item.Nodes[i].String() + ":"
		near := item.Edges[*item.Nodes[i]]

		for j := 0; j < len(near); j++ {
			s += near[j].String() + "->"
		}
		s += "\n"
	}
	fmt.Println(s)
}

// BFS 图的广度优先算法
func (item *ItemGraph) BFS() {

}
