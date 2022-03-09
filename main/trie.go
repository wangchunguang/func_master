package main

import "container/list"

// Trie 前缀树
type Trie struct {
	// 是否是某个单词的结尾
	IsTrie bool
	// 表示当前字符下一个字符有哪些
	ListTrie [26]*Trie
}

func Constructor_Trie() Trie {
	return Trie{}
}

func (this *Trie) Insert(word string) {

	for _, value := range word {
		if this.ListTrie[value-'a'] == nil { //判断第一层是否有这个数据
			this.ListTrie[value-'a'] = &Trie{}
		}
		// 向下面一层查找
		this = this.ListTrie[value-'a']
	}
	// 当一个单词结束的时候指定这个的结尾
	this.IsTrie = true
}

func (this *Trie) Search(word string) bool {
	for _, value := range word {
		// 当前层判断是否有这个数据 如果没有直接返回
		if this.ListTrie[value-'a'] == nil {
			return false
		}
		// 进行下一层寻找
		this = this.ListTrie[value-'a']
	}
	// 判断是不是结尾
	return this.IsTrie
}

func (this *Trie) StartsWith(prefix string) bool {
	for _, value := range prefix {
		if this.ListTrie[value-'a'] == nil {
			return false
		}
		this = this.ListTrie[value-'a']
	}
	return true
}

// UF 并查集的实体
type UF struct {
	union []int
}

// 并查集数组里面包含的是当前数的上一个节点，开始默认的是包含自己本身
// 初始化并查集
func newUF(cap int) *UF {
	uf := UF{
		make([]int, cap),
	}
	for i := 0; i < cap; i++ {
		uf.union[i] = i
	}
	return &uf
}

// Union 有序表将两个单独的团队合并
func (u *UF) Union(x, y int) {
	rootX := u.find(x)
	rootY := u.find(y)
	if rootY == rootX {
		return
	}
	// 将y的根节点变为x的上一个节点
	u.union[rootX] = rootY
}

// Connected 判断两个是不是同一个根节点
func (u *UF) Connected(x, y int) bool {
	return u.find(x) == u.find(y)
}

// 获取根结点
func (u *UF) find(x int) int {
	root := x
	// 获取根节点 ，因为根节点的指向的节点是它自己
	for u.union[root] != root {
		root = u.union[root]
	}
	// 如果开始查找根节点需要很多步，那么就替换当前节点的上一个节点为根节点
	for x != root {
		tmp := u.union[x]
		// 将查找节点所有的上序节点都改为指向根节点
		u.union[x] = root
		x = tmp
	}
	return root
}

type entry struct {
	key, value int
}

// LRUCache LRU缓存刷新策略
type LRUCache struct {
	cap   int
	cache map[int]*list.Element
	lst   *list.List
}

func Constructor_LRU(capacity int) LRUCache {
	return LRUCache{capacity, map[int]*list.Element{}, list.New()}
}

func (c *LRUCache) Get(key int) int {
	e := c.cache[key]
	if e == nil {
		return -1
	}
	c.lst.MoveToFront(e) // 刷新缓存使用时间
	return e.Value.(entry).value
}

func (c *LRUCache) Put(key, value int) {
	if e := c.cache[key]; e != nil {
		e.Value = entry{key, value}
		c.lst.MoveToFront(e) // 刷新缓存使用时间
		return
	}
	c.cache[key] = c.lst.PushFront(entry{key, value})
	if len(c.cache) > c.cap {
		delete(c.cache, c.lst.Remove(c.lst.Back()).(entry).key)
	}
}

type WordDictionary struct {
	listWord [26]*WordDictionary
	IsLeaf   bool
}

func Constructor() WordDictionary {
	return WordDictionary{}
}

func (this *WordDictionary) AddWord(word string) {
	//	 添加数据
	for _, value := range word {
		// 该字符不存在
		if this.listWord[value-'a'] == nil {
			this.listWord[value-'a'] = &WordDictionary{}
		}
		this = this.listWord[value-'a']
	}
	this.IsLeaf = true
}

func (this *WordDictionary) Search(word string) bool {
	//	 i表示当前比较到了第几位 node 表示当前第几位上面的数据
	var dfs func(i int, node *WordDictionary) bool
	dfs = func(i int, node *WordDictionary) bool {
		// base case
		if i == len(word) {
			return node.IsLeaf
		}
		//	 如果等于。的时候
		if word[i] == '.' {
			// 如果为点 啧遍历当前值下面的所有的前缀树，去递归处理
			for _, value := range node.listWord {
				if value != nil && dfs(i+1, value) {
					return true
				}
			}

		} else {
			// 不等于。的时候

			if node.listWord[word[i]-'a'] == nil {
				return false
			}
			return dfs(i+1, node.listWord[word[i]-'a'])
		}
		return false
	}
	return dfs(0, this)
}
