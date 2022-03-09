package num7

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

// 368. 最大整除子集
func largestDivisibleSubset(nums []int) []int {
	// 先将数组排序
	lenght := len(nums)
	sort.Ints(nums)
	// 因为发现的规律是，如果能判断nums[i] 整除nums[j] 那么nums[i]的最长子数组就是nums【j】的数量+1
	// 设置一个arr数组保存nums每个位置的最长子数组
	// 设置一个cur数组表示对应的i是从那个下标j动态转移过来的
	// 因为重复子问题 无后效性 所以可以采用动态规划
	arr := make([]int, lenght)
	list := make([]int, lenght)
	for i := 0; i < lenght; i++ {
		//	 设置num 表示当前i位置的最大长度
		//	因为没有进行判断的时候 数组的最大长度就是本身 所以默认值为1
		num := 1
		//	设置当前num是从什么位置转移过来的 ，如果当前i对前面的任何数组都没有整除，所以也就是i
		prev := i
		for j := i - 1; j >= 0; j-- {
			if nums[i]%nums[j] == 0 {
				//	 判断是否有更长的长度
				if arr[j]+1 > num {
					//	将num prev重新赋值
					num = arr[j] + 1
					prev = j
				}
			}
		}
		arr[i] = num
		list[i] = prev
	}
	fmt.Println(arr)
	fmt.Println(list)
	// 获取当前数组中最大的长度
	sum := -1
	// 表示当前最大的长度是从哪个位置转移过来的
	ids := -1
	for key, val := range arr {
		if val >= sum {
			sum = val
			ids = key
		}
	}
	// 创建一个数组用于返回
	res := []int{}
	// 将数据组装
	for len(res) != sum {
		res = append(res, nums[ids])
		ids = list[ids]
	}
	sort.Ints(res)
	return res
}

//367. 有效的完全平方数
func isPerfectSquare(num int) bool {
	if num == 1 {
		return true
	}
	i := 2
	for {
		sun := i * i
		if sun == num {
			return true
		}
		if sun > num {
			return false
		}
		i++
	}
}

// 365. 水壶问题
func canMeasureWater(x int, y int, targetCapacity int) bool {
	//	 因为有很多的可能性 所以采用bfs算法
	if targetCapacity == 0 {
		return false
	}
	if x+y < targetCapacity {
		return false
	}
	if x+y == targetCapacity {
		return true
	}
	return canMeasureWater_bfs(x, y, targetCapacity)
}

type State struct {
	X int
	Y int
}

func canMeasureWater_bfs(x, y, z int) bool {
	//	 bfs 套路 初始化队列 还要一个map记录是否计算过
	//	 循环判断队列长度是否为空
	//	 循环内，队首元素先出栈，判断是否访问过，防止陷入死循环，
	//	 判断截止条件 队列为空， 当前满足条件
	bfs_map := make(map[State]bool)
	init := State{
		X: 0,
		Y: 0,
	}
	queue := []State{init}
	for len(queue) > 0 {
		newStart := queue[0]
		queue = queue[1:]
		if bfs_map[newStart] {
			continue
		}
		bfs_map[newStart] = true
		//	 截止条件
		if newStart.X == z || newStart.Y == z || newStart.X+newStart.Y == z {
			return true
		}
		//	 下一次的装塔斯 倒满x
		queue = append(queue, State{x, newStart.Y})
		// 下一次操作 倒满y
		queue = append(queue, State{newStart.X, y})
		//	 下一次操作 清空x
		queue = append(queue, State{0, newStart.Y})
		// 下一次操作 清空y
		queue = append(queue, State{newStart.X, 0})
		//	 将x的水倒入y
		// 分为两种情况
		// 将x中的水倒入y中 还有剩余的水
		// y-newStart.Y 表示y杯子中还可以放入的水量
		if newStart.X > y-newStart.Y {
			queue = append(queue, State{newStart.X - (y - newStart.Y), y})
		} else {
			//	 将x的水倒入y中 没有剩余的水
			queue = append(queue, State{0, newStart.Y + newStart.X})
		}
		//	 y中的水倒入x
		if newStart.Y > x-newStart.X {
			queue = append(queue, State{x, newStart.Y - (x - newStart.X)})
		} else {
			queue = append(queue, State{newStart.Y + newStart.X, 0})
		}
	}
	return false
}

//363. 矩形区域不超过 K 的最大数值和
func maxSumSubmatrix(matrix [][]int, k int) int {
	// 计算矩形每一行依次计算的总和
	row, list := len(matrix), len(matrix[0])
	for i := 1; i < row; i++ {
		for j := 0; j < list; j++ {
			matrix[i][j] = matrix[i][j] + matrix[i-1][j]
		}
	}
	num := math.MinInt64
	for i := 0; i < row; i++ {
		for j := i; j < row; j++ {
			for q := 0; q < list; q++ {
				sum := 0
				for p := q; p >= 0; p-- {
					// 获取当前列的数据
					if i == 0 {
						sum += matrix[j][p]
					} else {
						sum += matrix[j][p] - matrix[i-1][p]
					}
					if sum <= k && sum >= num {
						num = sum
					}
				}
			}
		}
	}
	// 先固定上下边界 然后固定左右边界
	return num
}

// 357. 计算各个位数不同的数字个数
func countNumbersWithUniqueDigits(n int) int {
	// 因为当前位置的都会加上上一个位置的数
	if n == 0 {
		return 0
	}
	dp := make([]int, n)
	dp[0] = 1
	dp[1] = 10
	//dp[i]=dp[i-1]+(dp[i-1]-dp[i-2])*(10-(i-1));
	//加上dp[i-1]没什么可说的，加上之前的数字
	//dp[i-1]-dp[i-2]的意思是我们上一次较上上一次多出来的各位不重复的数字。以n=3为例，n=2已经计算了0-99之间不重复的数字了，我们需要判断的是100-999之间不重复的数字，那也就只能用10-99之间的不重复的数去组成三位数，而不能使用0-9之间的不重复的数，因为他们也组成不了3位数。而10-99之间不重复的数等于dp[2]-dp[1]。
	//当i=2时，说明之前选取的数字只有
	//1位，那么我们只要与这一位不重复即可，所以其实有9(10-1)种情况（比如1，后面可以跟0,2,3,4,5,6,7,8,9）。
	//当i=3时，说明之前选取的数字有2位，那么我们需要与2位不重复，所以剩余的
	//有8（10-2）种（比如12，后面可以跟0,3,4,5,6,7,8,9）
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + (dp[i-1]-dp[i-2])*(10-(i-1))
	}
	return dp[n]
}

// Twitter 355. 设计推特
type Twitter struct {
	//	 关注的用户采用图的形式表示数据
	//	 采用图的形式
	TwitterUser map[int]map[int]int
	//	每个用户对应的推文id
	TwitterId map[int]*Heap
}

func Constructor() Twitter {
	return Twitter{
		TwitterUser: map[int]map[int]int{},
		TwitterId:   map[int]*Heap{},
	}
}

func (this *Twitter) PostTweet(userId int, tweetId int) {
	//	 判断是否有数据
	val, ok := this.TwitterId[userId]
	user := &UserTwitter{unix: time.Now().Nanosecond(), tweetId: tweetId}
	//	 当不存在的数后
	if ok {
		val.PushMax(user)
		this.TwitterId[userId] = val
	} else {
		if _, flag := this.TwitterUser[userId]; flag {
			heap := NewHeap()
			heap.PushMax(user)
			this.TwitterId[userId] = heap
		} else {
			m := map[int]int{}
			m[userId] = userId
			this.TwitterUser[userId] = m
			heap := NewHeap()
			heap.PushMax(user)
			this.TwitterId[userId] = heap
		}

	}
}

func (this *Twitter) GetNewsFeed(userId int) []int {
	twitter_data := []*UserTwitter{}
	// 获取所有关注的信息
	for _, val := range this.TwitterUser[userId] {
		// 获取关注的用户id
		//	 通过用户id获取发布的信息
		heap := this.TwitterId[val]
		if heap == nil {
			continue
		}
		list := []*UserTwitter{}
		for i := 0; i < 10; i++ {
			delMax := heap.DelMax()
			if delMax == nil {
				break
			}
			twitter_data = append(twitter_data, delMax)
			list = append(list, delMax)
		}
		heap.PushListMax(list)
	}
	sort.Slice(twitter_data, func(i, j int) bool {
		return twitter_data[i].unix >= twitter_data[j].unix
	})
	arr := []int{}
	end := len(twitter_data)
	if end > 10 {
		end = 10
	}
	for i := 0; i < end; i++ {
		arr = append(arr, twitter_data[i].tweetId)
	}
	return arr
}

func (this *Twitter) Follow(followerId int, followeeId int) {
	val, ok := this.TwitterUser[followerId]
	if ok {
		val[followeeId] = followeeId
		this.TwitterUser[followerId] = val
	} else {
		m := map[int]int{}
		m[followeeId] = followeeId
		m[followerId] = followerId
		this.TwitterUser[followerId] = m
	}
}

func (this *Twitter) Unfollow(followerId int, followeeId int) {
	val, ok := this.TwitterUser[followerId]
	if !ok {
		return
	}
	for _, value := range val {
		if value == followeeId {
			delete(val, followeeId)
			this.TwitterUser[followerId] = val
			return
		}
	}
}

type Heap struct {
	data  []*UserTwitter
	count int
}
type UserTwitter struct {
	unix    int
	tweetId int
}

// NewHeap 初始化堆，第一个元素不用
func NewHeap() *Heap {
	// 第一个位置空着，不使用
	user := &UserTwitter{1, 1}
	data := []*UserTwitter{user}
	return &Heap{data: data, count: 0}
}

// Parent 父节点的位置
func (h *Heap) Parent(root int) int {
	return root / 2
}

// 左子树的位置
func (h *Heap) left(root int) int {
	return root * 2
}

// 右子树的位置
func (h *Heap) right(root int) int {
	return root*2 + 1
}

// Pop 返回堆顶元素
func (h *Heap) Pop() *UserTwitter {
	return h.data[1]
}

// 堆化的时候交换元素
func (h *Heap) exchange(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *Heap) Count() int {
	return h.count
}

// PushMax 新增元素 大根堆
func (h *Heap) PushMax(v *UserTwitter) {
	h.count++
	h.data = append(h.data, v)
	// 元素新增之后进行堆化
	h.swimMax(h.count)
}

// PushListMax 在大根堆里面新增多个元素
func (h *Heap) PushListMax(arr []*UserTwitter) {
	for _, value := range arr {
		h.PushMax(value)
	}
}

// DelMax 删除大根堆的头部元素
func (h *Heap) DelMax() *UserTwitter {
	if h.count < 1 {
		return nil
	}
	heapMax := h.data[1]
	// 将最后一个元素放到第一个堆顶位置，然后向下进行堆化
	h.exchange(1, h.count)
	h.data = h.data[:h.count]
	h.count--
	h.sinkMax(1)
	return heapMax
}

// 向上堆化 大根堆
func (h *Heap) swimMax(key int) {
	// 根节点<当前节点
	// 堆化
	for key > 1 && h.data[key].unix > h.data[h.Parent(key)].unix {
		h.exchange(h.Parent(key), key)
		key = h.Parent(key)
	}
}

//向下进行堆化
func (h *Heap) sinkMax(key int) {
	// 下沉到堆底
	for h.left(key) <= h.count {
		order := h.left(key)
		if h.right(key) <= h.count && h.data[order].unix < h.data[h.right(key)].unix {
			order = h.right(key)
		}
		// 节点比两个子节点都大,就不必下沉了
		if h.data[order].unix < h.data[key].unix {
			break
		}
		h.exchange(key, order)
		key = order
	}
}

// 354. 俄罗斯套娃信封问题
func maxEnvelopes(envelopes [][]int) int {
	dp := make([]int, len(envelopes))
	sort.Slice(envelopes, func(i, j int) bool {
		return envelopes[i][0] < envelopes[j][0]
	})
	ans := 1
	for i := 0; i < len(envelopes); i++ {
		//	 每个默认都是有一个套娃的
		dp[i] = 1
		for j := 0; j < i; j++ {
			if isEnvelopes(envelopes[j], envelopes[i]) {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
		ans = max(ans, dp[i])
	}
	return ans
}

func isEnvelopes(a, b []int) bool {
	if b[0] > a[0] && b[1] > a[1] {
		return true
	}
	return false
}

// SummaryRanges 352. 将数据流变为多个不相交区间
type SummaryRanges struct {
	Arr []bool
}

func Constructor_352() SummaryRanges {
	return SummaryRanges{
		Arr: make([]bool, 10001),
	}
}

func (this *SummaryRanges) AddNum(val int) {
	this.Arr[val] = true
}

func (this *SummaryRanges) GetIntervals() [][]int {
	lenght := len(this.Arr)
	if lenght == 0 {
		return nil
	}
	arr := [][]int{}
	start, end := -1, -1
	for i := 0; i < 10001; i++ {
		// 表示当前数据出现过
		if this.Arr[i] {
			// 判断有没有初始化开始位置与结束位置
			// 没有初始化开始位置和结束位置 ，那么开始位置和结束位置都是当前位置
			if start == -1 {
				start = i
				end = i
			} else {
				end = i
			}
		} else {
			// 当前位置没有出现，并且经历过初始化，那么将数据加入
			if start != -1 {
				arr = append(arr, []int{start, end})
				start = -1
				end = -1
			}
		}
	}
	// 最后判断最后一个数
	if start != -1 {
		arr = append(arr, []int{start, end})
	}
	return arr
}

// 350. 两个数组的交集 II
func intersect(nums1 []int, nums2 []int) []int {
	m := map[int]int{}
	arr := []int{}
	if len(nums1) < len(nums2) {
		tmp := nums1
		nums1 = nums2
		nums2 = tmp
	}
	for _, val := range nums1 {
		m[val]++
	}
	for _, val := range nums2 {
		if m[val] != 0 {
			arr = append(arr, val)
			m[val]--
		}
	}
	return arr
}

// 349. 两个数组的交集
func intersection(nums1 []int, nums2 []int) []int {
	m := make(map[int]bool)
	for _, val := range nums1 {
		m[val] = false
	}
	for _, val := range nums2 {
		if _, ok := m[val]; ok {
			m[val] = true
		}
	}

	arr := []int{}
	for key, val := range m {
		if val {
			arr = append(arr, key)
		}
	}
	return arr
}

// 347. 前 K 个高频元素
func topKFrequent(nums []int, k int) []int {
	m := map[int]int{}
	for _, val := range nums {
		m[val]++
	}
	arr := [][]int{}

	for key, value := range m {
		arr = append(arr, []int{value, key})
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i][0] >= arr[j][0]
	})
	list := make([]int, k)
	for i := 0; i < k; i++ {
		list[i] = arr[i][1]
	}
	return list
}

func reverseVowels(s string) string {

	start, end := 0, len(s)-1
	by := []byte(s)
	for start < end {
		for start < len(s) && !isVowels(by[start]) {
			start++
		}
		for end > 0 && !isVowels(by[end]) {
			end--
		}
		if start > end {
			break
		}
		by[start], by[end] = by[end], by[start]
		start++
		end--
	}
	return string(by)
}
func isVowels(by byte) bool {
	if by == 'a' || by == 'e' || by == 'i' || by == 'o' || by == 'u' ||
		by == 'A' || by == 'E' || by == 'I' || by == 'O' || by == 'U' {
		return true
	}
	return false
}

// 344. 反转字符串
func reverseString(s []byte) {
	start, end := 0, len(s)-1
	for start < end {
		s[start], s[end] = s[end], s[start]
		start++
		end--
	}
}

// 343. 整数拆分
func integerBreak(n int) int {
	//	 采用动态规划
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 1
	for i := 3; i <= n; i++ {
		for j := 1; j < i-1; j++ {
			// 因为整数i要拆分成整数的最大的乘积
			// 按照j区划分成了 i i-j
			// 遍历所有可以选择 i-j 拆或者不拆
			// 不拆就是i-j  拆分就是dp[i-j] 就是对子问题dp[i-j]的调用
			dp[i] = max(dp[i], max(dp[i-j], i-j)*j)
		}
	}
	return dp[n]
}

// 342. 4的幂
func isPowerOfFour(n int) bool {
	if n == 1 {
		return true
	}
	num := 1
	for {
		num *= 4
		if num > n {
			return false
		}
		if num == n {
			return true
		}
	}

}

// 337. 打家劫舍 III
func rob(root *TreeNode) int {
	m := map[*TreeNode]int{}
	rob_337(root, m)
	return m[root]
}

func rob_337(root *TreeNode, m map[*TreeNode]int) int {
	//	 当二叉树为空 直接返回
	if root == nil {
		return 0
	}
	if val, ok := m[root]; ok {
		return val
	}
	//	 当前节点表示可以获取
	rootVal := root.Val
	if root.Left != nil {
		//	 可以获取的是左子树的右子树 左子树的左子树
		rootVal += rob_337(root.Left.Left, m) + rob_337(root.Left.Right, m)
	}
	//	 如果右子树不为空的话
	if root.Right != nil {
		rootVal += rob_337(root.Right.Left, m) + rob_337(root.Right.Right, m)
	}
	//	  不获取根节点 啧获取左右子节点
	num := rob_337(root.Left, m) + rob_337(root.Right, m)
	m[root] = max(num, rootVal)
	return m[root]
}

// 336. 回文对
func palindromePairs(words []string) [][]int {
	// 创建将数据存入m中，key表示相反的字符，如果在原数组中有相同的表示回文对 value 表示索引
	m1 := make(map[string]int)
	for key, val := range words {
		buffer := bytes.Buffer{}
		for i := len(val) - 1; i >= 0; i-- {
			buffer.WriteByte(val[i])
		}
		m1[buffer.String()] = key + 1
	}
	arr := [][]int{}
	for key, val := range words {
		//	 如果不等 选择较小长度的那个匹配长度匹配完成，剩余的字符串是回文串
		for j := 0; j <= len(val); j++ {
			left := val[:j]
			right := val[j:]
			// 因为回文找一个中心轴的话，他两边都是相等的
			if j != 0 && isPalindrome(left) && m1[right] != 0 && m1[right] != key+1 {
				arr = append(arr, []int{m1[right] - 1, key})
			}
			if isPalindrome(right) && m1[left] != 0 && m1[left] != key+1 {
				arr = append(arr, []int{key, m1[left] - 1})
			}
		}
	}
	return arr
}

// 判断是否是回文
func isPalindrome(str string) bool {
	left, right := 0, len(str)-1
	for left < right {
		if str[left] != str[right] {
			return false
		}
		left++
		right--
	}
	return true
}

// 334. 递增的三元子序列
func increasingTriplet(nums []int) bool {
	if len(nums) < 3 {
		return false
	}
	// a记录最小的数
	a := math.MaxInt64
	// b记录第二小的数
	b := math.MaxInt64

	for _, val := range nums {
		if val <= a {
			// 最小的数
			a = val
		} else if val <= b {
			// 第二小的数
			b = val
		} else {
			// 如果走到这一步 就直接返回成功
			return true
		}
	}
	return false
}

// 332. 重新安排行程
func findItinerary(tickets [][]string) []string {
	d_map := map[string][]string{}
	for _, v := range tickets {
		d_map[v[0]] = append(d_map[v[0]], v[1])
	}
	//	 将每条数据进行排序
	for _, val := range d_map {
		sort.Strings(val)
	}
	ans := []string{}
	findItinerary_dfs(&ans, "JFK", d_map)
	return ans
}

func findItinerary_dfs(ans *[]string, path string, d_map map[string][]string) {
	for len(d_map[path]) > 0 {
		v := d_map[path][0]
		d_map[path] = d_map[path][1:]
		findItinerary_dfs(ans, v, d_map)
	}
	*ans = append([]string{path}, *ans...)
}

// 331. 验证二叉树的前序序列化
func isValidSerialization(preorder string) bool {
	//第一种，采用栈的形式
	// 如果是叶子节点，就是两个#号，然后抵消，之后将他的父节点也变为星号
	// 最后只有一个
	if preorder == "#" {
		return true
	}
	stack := []string{}
	split := strings.Split(preorder, ",")
	for _, val := range split {
		// 当前的这个字符不是#
		stack = append(stack, val)
		for len(stack) >= 3 && stack[len(stack)-1] == "#" && stack[len(stack)-2] == "#" && stack[len(stack)-3] != "#" {
			stack = stack[:len(stack)-3]
			stack = append(stack, "#")
		}
	}
	return len(stack) == 1 && stack[0] == "#"

	// 第二种方式 采用入度出度的形式
	//split := strings.Split(preorder, ",")
	//// 入度默认为1
	//diff := 1
	//for _, val := range split {
	//	// 每次的入度都为1 所以减一，头结点也是，因为初始化为1了
	//	diff--
	//	if diff < 0 {
	//		return false
	//	}
	//	if val != "#"  {
	//		// 出度的话就直接+2
	//		diff += 2
	//	}
	//}
	//return diff ==0
}

//  330. 按要求补齐数组
func minPatches(nums []int, n int) int {
	//	 累加的总和
	total := 0
	//	 需要补充数字的个数
	count := 0
	//	 访问的下标
	index := 0
	for total < n {
		if index < len(nums) && nums[index] <= total+1 {

			total += nums[index]
			index++
		} else {
			total = total + (total + 1)
			count++
		}
	}
	return count
}

// 329. 矩阵中的最长递增路径
func longestIncreasingPath(matrix [][]int) int {
	arr := make([][]int, len(matrix))
	// 定义备忘录
	for i := 0; i < len(matrix); i++ {
		arr[i] = make([]int, len(matrix[0]))
	}
	num := 1

	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			//	 前后左右一次走
			dfs := longestIncreasingPath_dfs(matrix, arr, i, j)
			num = max(dfs, num)
		}
	}
	return num
}

func longestIncreasingPath_dfs(matrix [][]int, arr [][]int, i, j int) int {
	//	 如果当前走过的路径为true
	if arr[i][j] != 0 {
		return arr[i][j]
	}
	// 当前位置默认为1
	num := 1
	//  进行上下左右进行判断
	if i-1 >= 0 && matrix[i-1][j] > matrix[i][j] {
		num = max(num, longestIncreasingPath_dfs(matrix, arr, i-1, j)+1)
	}
	if i+1 < len(matrix) && matrix[i+1][j] > matrix[i][j] {
		num = max(num, longestIncreasingPath_dfs(matrix, arr, i+1, j)+1)
	}
	if j+1 < len(matrix[0]) && matrix[i][j+1] > matrix[i][j] {
		num = max(num, longestIncreasingPath_dfs(matrix, arr, i, j+1)+1)
	}
	if j-1 >= 0 && matrix[i][j-1] > matrix[i][j] {
		num = max(num, longestIncreasingPath_dfs(matrix, arr, i, j-1)+1)
	}
	arr[i][j] = num
	return num
}

// 328. 奇偶链表
func oddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// 定义奇数的头结点
	odd := head
	// 定义偶数的头结点
	even := head.Next
	evenHead := even
	for even != nil && even.Next != nil {
		// 确定奇数的下一个节点就是偶数的下一个节点
		odd.Next = even.Next
		// 将奇数的变为下一个节点
		odd = odd.Next
		// 偶数的下一个节点就是奇数的下一个节点
		even.Next = odd.Next
		// 将偶数置为下一个节点
		even = even.Next
	}
	// 奇数后面连接的是偶数的下一个节点
	odd.Next = evenHead
	return head
}

// 326. 3的幂
func isPowerOfThree(n int) bool {
	if n == 1 {
		return true
	}
	num := 1
	for {
		num *= 3
		if n == num {
			return true
		}
		if num > n {
			return false
		}
	}
}

// 324. 摆动排序 II
func wiggleSort(nums []int) {
	lenght := len(nums)
	sort.Ints(nums)
	arr := make([]int, lenght)

	p := lenght - 1
	for i := 1; i < lenght; i += 2 {
		arr[i] = nums[p]
		p--
	}
	for i := 0; i < lenght; i += 2 {
		arr[i] = nums[p]
		p--
	}
	copy(nums, arr)
}

//322. 零钱兑换
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
	}
	for i := 0; i <= amount; i++ {
		for _, val := range coins {
			if i >= val {
				dp[i] = min(dp[i], dp[i-val]+1)
			}
		}
	}
	if dp[amount] == amount+1 {
		return -1
	}
	return dp[amount]
}

// 两个数组剩余的数据的长度刚好等于k 则直接返回
func maxNumber_321(nums1 []int, nums2 []int, k int) []int {
	arr := []int{}
	m, n := len(nums1), len(nums2)
	if m+n == k {
		arr = append(nums1, nums2...)
	}
	return arr
}

// 319. 灯泡开关
func bulbSwitch(n int) int {

	return int(math.Sqrt(float64(n)))
}

// 318. 最大单词长度乘积
func maxProduct(words []string) int {
	sum := 0
	for key, value := range words {
		//	 定义map封装数据
		m_val := make(map[int32]bool)
		// 将最长的数据放入里面
		for _, val := range value {
			m_val[val] = true
		}
		for i := key + 1; i < len(words); i++ {
			flag := true
			for _, v := range words[i] {
				if m_val[v] {
					flag = false
					break
				}
			}
			if flag {
				sum = max(sum, len(value)*len(words[i]))
			}

		}
	}
	return sum
}

// 316. 去除重复字母
func removeDuplicateLetters(s string) string {
	// 每个元素出现的个数
	count := make([]int, 26)
	// 判断是否在栈中
	exist := make([]bool, 26)
	// 单调栈
	stack := []rune{}
	for _, v := range s {
		count[v-'a']++
	}
	for _, val := range s {
		//	 如果栈中已经存在该数，跳过
		if exist[val-'a'] {
			//	 减少这个数出现的次数
			count[val-'a']--
			continue
		}
		//	 出栈的核心
		//栈内有元素
		// 栈顶元素大于当前元素
		//栈顶元素在后续有出现
		for len(stack) > 0 && stack[len(stack)-1] > val && count[stack[len(stack)-1]-'a'] > 0 {
			//	 将数据出栈，并且将元素修改为未出现过
			exist[stack[len(stack)-1]-'a'] = false
			//	删除栈顶元素
			stack = stack[:len(stack)-1]

		}
		//	 将新的字符进栈
		stack = append(stack, val)
		exist[val-'a'] = true
		count[val-'a']--

	}
	return string(stack)
}

// 315. 计算右侧小于当前元素的个数
func countSmaller(nums []int) []int {
	node := &TreeNode{Val: nums[len(nums)-1]}
	count := make([]int, len(nums))
	for i := len(nums) - 2; i >= 0; i-- {
		var data int
		InsertNode(node, &TreeNode{Val: nums[i]}, &data)
		count[i] = data
	}
	return count
}

// 313. 超级丑数
func nthSuperUglyNumber(n int, primes []int) int {
	//	 计算前n个丑数
	dp := make([]int, n)
	dp[0] = 1
	// 选择每个质数因子当前需要进行到那一步
	indexes := make([]int, len(primes))
	for i := 1; i < n; i++ {
		// 因 为有最小值 先假设一个最大值
		dp[i] = math.MaxInt64
		for j := 0; j < len(primes); j++ {
			dp[i] = min(dp[i], dp[indexes[j]]*primes[j])
		}
		// dp[i] 是之前的哪个丑数乘以对应的 primes[j] 选出来的，给它加 1
		for j := 0; j < len(primes); j++ {
			if dp[i] == dp[indexes[j]]*primes[j] {
				indexes[j]++
			}
		}
	}
	return dp[n-1]
}

// 312. 戳气球
func maxCoins(nums []int) int {
	n := len(nums)
	nums = append(nums, 1)
	nums = append([]int{1}, nums...)
	dp := make([][]int, len(nums))
	for i := 0; i < len(nums); i++ {
		dp[i] = make([]int, len(nums))
	}
	// l is the length of subarray. We start with l= 1, end with l = n.
	for l := 1; l <= n; l++ {
		// i is the start point in this subarray.
		for i := 1; i <= n-l+1; i++ {
			// j is the subarray's end.
			j := i + l - 1
			// k is the break point to separate.
			for k := i; k <= j; k++ {
				dp[i][j] = max(dp[i][j], dp[i][k-1]+nums[i-1]*nums[k]*nums[j+1]+dp[k+1][j])
			}
		}
	}
	return dp[1][n]
}

// 310. 最小高度树
func findMinHeightTrees(n int, edges [][]int) []int {
	if n == 1 {
		return []int{0}
	}
	if n == 2 {
		return []int{0, 1}
	}

	//建立邻接图和入度统计
	graph := make(map[int][]int, n)
	degree := make([]int, n)
	for i := range edges {
		u, v := edges[i][0], edges[i][1]
		graph[v] = append(graph[v], u)
		graph[u] = append(graph[u], v)
		degree[v]++
		degree[u]++
	}

	//将度为1的全部加入队列
	queue := []int{}
	for i := 0; i < n; i++ {
		if degree[i] == 1 {
			queue = append(queue, i)
		}
	}

	cnt := len(queue)
	for n > 2 { //当节点数小于等于2的时候停止，这里不理解可以画图
		n -= cnt      //n去掉所有队列中的节点(这些节点度为1)
		for cnt > 0 { //遍历队列

			tmp := queue[0]
			queue = queue[1:] //queue.pop()

			degree[tmp] = 0                        //删去当前节点
			for i := 0; i < len(graph[tmp]); i++ { //遍历当前节点的邻接节点
				if degree[graph[tmp][i]] != 0 {
					degree[graph[tmp][i]]--         //去掉与当前节点的关系
					if degree[graph[tmp][i]] == 1 { //如果度为1了 就加入队列
						queue = append(queue, graph[tmp][i])
					}
				}
			}
			cnt--
		}
		cnt = len(queue)
	}
	ans := []int{}
	for _, v := range queue {
		ans = append(ans, v)
	}
	return ans
}

// 309. 最佳买卖股票时机含冷冻期
func maxProfit(prices []int) int {
	lenght := len(prices)
	if lenght < 2 {
		return 0
	}
	dp := make([]int, 3)
	// 0表示 持有
	// 1 表示 未持有
	// 2 表示 冷静期
	//  表示持有
	dp[0] = -prices[0]
	// 表示未持有
	dp[1] = 0
	// 表示冷静期
	dp[2] = 0
	num1, num2, num3 := 0, 0, 0
	for i := 1; i < lenght; i++ {
		// 当天持有股票 表示前一天持有 或者当天买入，但是买入必须过了冷静期才可以买入
		// 买入表示减去当前的价钱
		num1 = max(dp[0], dp[2]-prices[i])
		// 当天未持有
		// 从上一天未持有 对比上一天持有 当天卖出
		num2 = max(dp[1], dp[0]+prices[i])
		// 当天冷静期
		// 冷静期是前一天的冷静期，或者前一天卖出股票才可能进入冷静期
		num3 = max(dp[2], dp[1])
		dp[0], dp[1], dp[2] = num1, num2, num3
	}
	num := max(dp[1], dp[2])
	return num
}

// NumArray 307. 区域和检索 - 数组可修改
type NumArray struct {
	seg *SegmentTree
}

func Constructor_307(nums []int) NumArray {
	return NumArray{seg: NewSegmentTree(nums)}
}

func (this *NumArray) Update(index int, val int) {
	this.seg.Update(index, val)
}

func (this *NumArray) SumRange(left int, right int) int {
	return this.seg.SunRange(left, right)
}

// 306. 累加数
func isAdditiveNumber(num string) bool {
	lenght := len(num)
	if lenght < 3 {
		return false
	}
	// 排除第一位是0的状态
	if num[0] == '0' {
		return false
	}
	for i := 1; i < lenght; i++ {
		// 第一个数
		lst, _ := strconv.Atoi(num[:i])
		// lst当前第一位数，i第二位数的开始位置
		if isAdditiveNumber_dfs(num, lst, i) {
			return true
		}
	}
	return false
}

func isAdditiveNumber_dfs(num string, lst, idx int) bool {
	n := len(num)
	isAdditive := false
	// 因为切片后面的不包括，所以是+1
	for nxtStart := idx + 1; nxtStart < n; nxtStart++ {
		if nxtStart > idx+1 && num[idx] == '0' { // 排除前导 0
			break
		}
		//	 第二个数
		cur, _ := strconv.Atoi(num[idx:nxtStart])
		// 计算第一个和第二个之和
		sum := strconv.Itoa(lst + cur)
		dur := len(sum)
		nextEnd := nxtStart + dur
		if nextEnd > n {
			break
		}
		//	 判断和是否相等
		if sum == num[nxtStart:nextEnd] {
			if nextEnd == n {
				return true
			}
			isAdditive = isAdditiveNumber_dfs(num, cur, nxtStart)
		}
	}
	return isAdditive
}

// NumMatrix 304. 二维区域和检索 - 矩阵不可变
type NumMatrix struct {
	Arr [][]int
}

func Constructor_304(matrix [][]int) NumMatrix {
	// 将每一列的累加和计算出来
	for i := 1; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			matrix[i][j] = matrix[i][j] + matrix[i-1][j]
		}
	}
	return NumMatrix{Arr: matrix}
}

func (this *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {
	num := 0
	if row1 == 0 {
		for i := col1; i <= col2; i++ {
			num += this.Arr[row2][i]
		}
	} else {
		for i := col1; i <= col2; i++ {
			num += this.Arr[row2][i] - this.Arr[row1-1][i]
		}
	}
	return num
}

// NumArray 303. 区域和检索 - 数组不可变
type NumArray_303 struct {
	Arr []int
}

func Constructor_303(nums []int) NumArray_303 {

	return NumArray_303{Arr: nums}
}

func (this *NumArray_303) SumRange(left int, right int) int {
	num := 0
	for i := left; i <= right; i++ {
		num += this.Arr[i]
	}
	return num
}

// 301. 删除无效的括号
func removeInvalidParentheses(s string) []string {
	lremove, rremove := 0, 0
	for _, val := range s {
		if val == '(' {
			// 遇见左括号+1
			lremove += 1
		} else if val == ')' && lremove == 0 {
			// 遇见右括号 但是左括号没有数据 左括号+1
			rremove += 1
		} else if val == ')' && lremove > 0 {
			// 遇见有括号，并且已经有左括号的数量，表示一对出现，左括号减一
			lremove -= 1
		}
	}
	arr := []string{}
	removeInvalidParentheses_dfs(lremove, rremove, 0, s, &arr)
	return arr
}

func removeInvalidParentheses_dfs(lremove, rremove, start int, s string, arr *[]string) {
	//	 如果当左右括号都删除为0之后判断是否达标，达标的话，将数据存储
	if lremove == 0 && rremove == 0 {
		if valid(s) {
			*arr = append(*arr, s)
		}
		return
	}
	for i := start; i < len(s); i++ {
		if i > start && s[i] == s[i-1] {
			continue
		}
		if s[i] == '(' && lremove > 0 {
			removeInvalidParentheses_dfs(lremove-1, rremove, i, s[:i]+s[i+1:], arr)
		}
		if s[i] == ')' && rremove > 0 {
			removeInvalidParentheses_dfs(lremove, rremove-1, i, s[:i]+s[i+1:], arr)
		}
	}

}

// 判断括号是否可以全部抵消
func valid(s string) bool {
	cnt := 0
	for _, val := range s {
		if val == '(' {
			cnt++
		} else if val == ')' {
			cnt--
			if cnt < 0 {
				return false
			}
		}
	}
	return cnt == 0
}
