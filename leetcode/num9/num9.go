package num9

import (
	"bytes"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

//496. 下一个更大元素 I
func nextGreaterElement(nums1 []int, nums2 []int) []int {
	//	 创建一个map  将第二个数组预处理
	m := make(map[int]int)
	start := 0
	for start < len(nums2) {
		flag := false
		for i := start + 1; i < len(nums2); i++ {
			if nums2[i] > nums2[start] {
				flag = true
				m[nums2[start]] = nums2[i]
				break
			}
		}
		if !flag {
			m[nums2[start]] = -1
		}
		start++
	}
	arr := make([]int, len(nums1))
	for key, val := range nums1 {
		arr[key] = m[val]
	}
	return arr
}

// 495. 提莫攻击
func findPoisonedDuration(timeSeries []int, duration int) int {
	sort.Ints(timeSeries)
	sum := 0
	for i := 1; i < len(timeSeries); i++ {
		if timeSeries[i]-timeSeries[i-1] >= duration {
			sum += duration
		} else {
			sum += timeSeries[i] - timeSeries[i-1]
		}
	}
	return sum + duration
}

// 494. 目标和
func findTargetSumWays(nums []int, target int) int {
	// dp
	// dp方程式 dp[i][j] = dp[i-1][j+nums[i] + dp[i-1][j-nums[j]]
	sum := 0
	for _, val := range nums {
		sum += val
	}
	if abs(target) > abs(sum) {
		return 0
	}
	dp := make([][]int, len(nums))
	for i := 0; i < len(dp); i++ {
		//	 因为每个都有两个形式 ，要么加上当前的数 要么减去当前的数
		dp[i] = make([]int, sum*2+1)
	}
	// 初始化第0位，因为可能 0-nums[0]小于0，所以就直接采用和减去第一个值，那么返回的数据也需要 加上sum
	dp[0][sum+nums[0]]++
	dp[0][sum-nums[0]]++
	for i := 1; i < len(nums); i++ {
		for j := 0; j < len(dp[i]); j++ {
			l, r := 0, 0
			if j-nums[i] >= 0 {
				l = dp[i-1][j-nums[i]]
			}
			if j+nums[i] < len(dp[i]) {
				r = dp[i-1][j+nums[i]]
			}
			dp[i][j] = l + r
		}
	}
	return dp[len(nums)-1][sum+target]
	// 深度优先算法
	var dfs func(start, sum int) int
	dfs = func(start, sum int) int {
		if start == len(nums) {
			if sum == target {
				return 1
			}
			return 0
		}
		return dfs(start+1, sum+nums[start]) + dfs(start+1, sum-nums[start])
	}
	return dfs(0, 0)
}

// 493. 翻转对
func reversePairs(nums []int) int {
	count := 0
	var reversePairsSort func(start, end int)
	reversePairsSort = func(start, end int) {
		if start == end {
			return
		}
		mid := start + (end-start)/2
		reversePairsSort(start, mid)
		reversePairsSort(mid+1, end)
		//	 此时左边和右边都是升序
		// 左边的开头
		i := start
		// 右边的开头
		j := mid + 1
		for i <= mid && j <= end {
			// 因为是升序，如果一个满足的话 ，那么就是 mid减去i的位置都满足
			// 比如 6,7,8,9,10    ,1,2,3,4 满足的话，1全部满足 2全部满足 3左边满足四个 4左边满足两个
			if nums[i] > 2*nums[j] {
				count += mid - i + 1
				j++
			} else {
				i++
			}
		}
		//	 创建辅助数组，存放合并排序的数
		tmep := make([]int, end-start+1)
		index := 0
		i = start
		j = mid + 1
		for i <= mid && j <= end {
			if nums[i] < nums[j] {
				tmep[index] = nums[i]
				index++
				i++
			} else {
				tmep[index] = nums[j]
				index++
				j++
			}
		}
		for i <= mid {
			tmep[index] = nums[i]
			index++
			i++
		}
		for j <= end {
			tmep[index] = nums[j]
			index++
			j++
		}
		k := 0
		for i := start; i <= end; i++ { // 根据合并后的情况，更新nums
			nums[i] = tmep[k]
			k++
		}

	}
	reversePairsSort(0, len(nums)-1)
	return count
}

// 492. 构造矩形
func constructRectangle(area int) []int {
	sqrt := int(math.Sqrt(float64(area)))
	for i := sqrt; i <= area; i++ {
		if area%i == 0 {
			if i > area/i {
				return []int{i, area / i}
			}
			return []int{area / i, i}
		}
	}
	return []int{}
}

// 491. 递增子序列
func findSubsequences(nums []int) [][]int {
	arr := [][]int{}
	m := map[string]bool{}
	var dfs func(i int, cur []int)
	dfs = func(i int, cur []int) {
		buffer := bytes.Buffer{}
		for k := 0; k < len(cur); k++ {
			buffer.WriteString(strconv.Itoa(cur[k]))
			buffer.WriteString("-")
		}
		if m[buffer.String()] {
			return
		}
		m[buffer.String()] = true
		if len(cur) >= 2 {
			per := make([]int, len(cur))
			copy(per, cur)
			arr = append(arr, per)
		}
		for j := i; j < len(nums); j++ {
			if len(cur) > 0 && nums[j] < cur[len(cur)-1] {
				continue
			}
			cur = append(cur, nums[j])
			dfs(j+1, cur)
			cur = cur[:len(cur)-1]
		}
	}
	dfs(0, []int{})

	return arr
}

type state struct {
	board string
	hand  [5]int
}

// 488. 祖玛游戏
func findMinStep(board string, hand string) int {
	cache := map[string]string{}
	COLORS := "RYBGW"
	var clean func(b string) string
	clean = func(board string) string {
		if v, ok := cache[board]; ok {
			return v
		}
		res := board
		for i, j := 0, 0; i < len(board); {
			for j < len(board) && board[i] == board[j] {
				j += 1
			}
			if j-i > 2 {
				res = clean(board[:i] + board[j:])
				cache[board] = res
				return res
			}
			i = j
		}
		cache[board] = res
		return res
	}
	cnts := func(hand string) [5]int {
		res := [5]int{}
		for i := 0; i < len(hand); i++ {
			for j, c := range COLORS {
				if hand[i] == byte(c) {
					res[j]++
					break
				}
			}
		}
		return res
	}
	queue := make([]state, 0, 6)
	init := state{board, cnts(hand)}
	queue = append(queue, init)
	visited := map[state]int{}
	visited[init] = 0
	for len(queue) > 0 {
		curState := queue[0]
		cur_board, cur_hand := curState.board, curState.hand
		if len(cur_board) == 0 {
			return visited[curState]
		}
		for i := 0; i <= len(cur_board); i++ {
			for j, r := range COLORS {
				if cur_hand[j] > 0 {
					c := byte(r)
					// 第 1 个剪枝条件: 只在连续相同颜色的球的开头位置插入新球(在它前面插入过了，不需要再插入，意义相同)
					if i > 0 && cur_board[i-1] == c {
						continue
					}

					/**
					 *  第 2 个剪枝条件: 只在以下两种情况放置新球
					 *  - 第 1 种情况 : 当前后颜色相同且与当前颜色不同时候放置球
					 *  - 第 2 种情况 : 当前球颜色与后面的球的颜色相同
					 */
					choose := false
					if 0 < i && i < len(cur_board) && cur_board[i-1] == cur_board[i] && cur_board[i-1] != c {
						choose = true
					}
					if i < len(cur_board) && cur_board[i] == c {
						choose = true
					}

					if choose {
						nxt := [5]int{}
						for k, _ := range COLORS {
							nxt[k] = cur_hand[k]
						}
						nxt[j] -= 1

						nextState := state{clean(cur_board[:i] + string(c) + cur_board[i:]), nxt}
						if _, ok := visited[nextState]; !ok {
							queue = append(queue, nextState)
							visited[nextState] = visited[curState] + 1
						}
					}
				}
			}
		}
	}
	return -1
}

// PredictTheWinner 486. 预测赢家
func PredictTheWinner(nums []int) bool {
	dp := make([][]int, len(nums))

	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(nums))
	}
	// dp[i][j] 表示的是这个在开始为i，结束为j的范围优先选择的数
	for i := 0; i < len(nums); i++ {
		dp[i][i] = nums[i]
	}
	for i := len(nums) - 2; i >= 0; i-- {
		for j := i + 1; j < len(nums); j++ {
			dp[i][j] = max(nums[i]-dp[i+1][j], nums[j]-dp[i][j-1])
		}
	}
	return dp[0][len(nums)-1] >= 0
	m := make(map[string]int)
	// 博弈论
	var dfs func(start, end int) int
	dfs = func(start, end int) int {
		//	 如果相等的时候 直接返回相等的数字
		key := strconv.Itoa(start) + "-" + strconv.Itoa(end)
		if val, ok := m[key]; ok {
			return val
		}
		if start == end {
			m[key] = nums[start]
			return m[key]
		}
		//	 当自己选择第一个 对手只能选择第二个到最后一个
		num1 := nums[start] - dfs(start+1, end)
		// 当自己选择最后一个，对手只能选择第一个到倒数第二个
		nums2 := nums[end] - dfs(start, end-1)
		// 选择结束之后，就是判断选择谁大 返回谁
		if num1 > nums2 {
			m[key] = num1
			return m[key]
		}
		m[key] = nums2
		return m[key]
	}
	return dfs(0, len(nums)-1) >= 0
}

// 485. 最大连续 1 的个数
func findMaxConsecutiveOnes(nums []int) int {

	sum := 0
	num := 0
	start, end := 0, 0
	for end < len(nums) {
		if nums[end] == 1 {
			end++
		} else {
			num = end - start
			sum = max(num, sum)
			end++
			start = end
		}

	}

	sum = max(sum, end-start)
	return sum
}

// 482. 密钥格式化
func licenseKeyFormatting(s string, k int) string {
	split := strings.Split(s, "-")
	buffer := bytes.Buffer{}
	for _, val := range split {
		buffer.WriteString(val)
	}
	upper := buffer.String()
	s2 := strings.ToUpper(upper)
	if len(s2) <= k {
		return s2
	}
	i := 1
	for i*k < len(s2) {
		i++
	}
	i--
	cur := len(s2) - i*k
	b := bytes.Buffer{}
	b.WriteString(s2[:cur])
	for cur < len(s2) {
		b.WriteString("-")
		b.WriteString(s2[cur : cur+k])
		cur += k
	}
	return b.String()
}

// 481. 神奇字符串
func magicalString(n int) int {
	//1221121221221121122
	// 找规律
	//	str的构造就是 1 和2 交替插入
	//	str的构造：
	//	 index = 0，str = “”，尾部添加一个'1'，str更新为“1”
	//	 index = 1，str = “1”，尾部添加str[index] - '0' = 2个 ‘2’，str 更新为 “122”，
	//	 index = 2，str = “122”，尾部添加str[index] - '0' = 2个 ‘1’，str 更新为 “122 11”，
	//	 index = 3，str = “12211”，尾部添加str[index] - '0' = 1个 ‘2’，str更新为“12211 2”
	//	 index = 4，str = “122112”，尾部添加str[index] - '0' = 1个 ‘1’，str更新 “122112 1”，
	//	 index = 5，str = “1221121”，尾部添加str[index] - '0' = 2个 ‘2’，str更新为“1221121 22”
	//	 index = 6，str = “122112122”，尾部添加str[index] - '0' = 1个‘1’，str 更新 “122112122 1”，
	//	 index = 7，str = “1221121221”，尾部添加str[index] - '0' = 2个‘2’，str更新为“1221121221 22”
	by := make([]byte, 0)
	by = append(by, '1')
	by = append(by, '2')
	by = append(by, '2')
	flag := true
	for i := 2; i < n && len(by) <= n; i++ {
		num, _ := strconv.Atoi(string(by[i]))
		if flag {
			if num-0 == 1 {
				by = append(by, '1')
			} else {
				by = append(by, '1')
				by = append(by, '1')
			}
			flag = false
		} else {
			if num-0 == 1 {
				by = append(by, '2')
			} else {
				by = append(by, '2')
				by = append(by, '2')
			}
			flag = true
		}
	}
	count := 0
	for i := 0; i < n; i++ {
		if by[i] == '1' {
			count++
		}
	}

	return count
}

// 480. 滑动窗口中位数
func medianSlidingWindow(nums []int, k int) []float64 {
	arr := []float64{}
	lenght := len(nums)
	list := []int{}
	list = append(list, nums[0:k]...)
	sort.Ints(list)
	if k%2 == 1 {
		arr = append(arr, float64(list[k/2]))
	} else {
		var num float64
		num = float64(list[(k/2)-1]) + float64(list[k/2])
		arr = append(arr, num/2)
	}
	l, r := 0, k
	for r < lenght {
		//先将l的数据清除，
		left := sort.Search(len(list), func(i int) bool {
			return nums[l] <= list[i]
		})
		list = append(list[:left], list[left+1:]...)
		right := sort.Search(len(list), func(i int) bool {
			return nums[r] < list[i]
		})
		list = append(list[:right], append([]int{nums[r]}, list[right:]...)...)
		if k%2 == 1 {
			arr = append(arr, float64(list[k/2]))
		} else {
			var num float64
			num = float64(list[(k/2)-1]) + float64(list[k/2])
			arr = append(arr, num/2)
		}
		l++
		r++
	}
	return arr
}

// 479. 最大回文数乘积
/*
	参考大佬们的解法：
	从大到小构造一个回文数，看这个回文数能不能由给定的数字相乘得到（也是从大到小枚举每个数字，越大的数相乘得到的数也就越大）
*/
func largestPalindrome(n int) int {
	if n == 1 {
		return 9
	}

	num := math.Pow10(n) - 1
	for i := num; i > 0; i-- {
		s1 := strconv.Itoa(int(i))
		sum, _ := strconv.Atoi(s1 + reverse([]rune(s1)))
		for x := int(num); x*x >= sum; x-- {
			if sum%x == 0 {
				return sum % 1337
			}
		}
	}
	return -1
}

func reverse(s []rune) string {
	l, r := 0, len(s)-1
	for l < r {
		s[l], s[r] = s[r], s[l]
		l++
		r--
	}
	return string(s)
}

//478. 在圆内随机生成点
type Solution struct {
	//	 正方形的四条边
	X, Y, Radius float64
}

func Constructor_478(radius float64, x_center float64, y_center float64) Solution {
	return Solution{
		X:      x_center,
		Y:      y_center,
		Radius: radius,
	}
}

func (s *Solution) RandPoint() []float64 {
	for {

		a := s.X - s.Radius + rand.Float64()*2*s.Radius
		b := s.Y - s.Radius + rand.Float64()*2*s.Radius

		dx := a - s.X
		dy := b - s.Y

		if dx*dx+dy*dy < s.Radius*s.Radius {
			return []float64{a, b}
		}
	}
}

// 475. 供暖器
func findRadius(houses []int, heaters []int) int {
	res := 0
	return res
}

// 474. 一和零
func findMaxForm(strs []string, m int, n int) int {
	dp := make([][]int, m+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, n+1)
	}
	for _, val := range strs {
		zero, one := strings.Count(val, "0"), strings.Count(val, "1")
		//	采用0 1背包算法，
		//最需要计算的值到当前字符串计算出来的数，依次相减，最后得出的结论
		for k := m; k >= zero; k-- {
			for j := n; j >= one; j-- {
				dp[k][j] = max(dp[k][j], dp[k-zero][j-one]+1)
			}
		}
	}
	return dp[m][n]
}

// 473. 火柴拼正方形
func makesquare(matchsticks []int) bool {
	sum := 0
	for _, val := range matchsticks {
		sum += val
	}
	// 判断是否四条边都满足
	if sum%4 != 0 {
		return false
	}
	sort.Slice(matchsticks, func(i, j int) bool {
		return matchsticks[i] > matchsticks[j]
	})
	// 计算平均值
	svg := sum / 4
	arr := make([]int, 4)
	// 采用回溯算法
	var dfs func(start int) bool
	dfs = func(start int) bool {
		if start >= len(matchsticks) {
			return arr[0] == arr[1] && arr[1] == arr[2] && arr[2] == arr[3]
		}
		// 因为是四条边
		for i := 0; i < 4; i++ {
			num := arr[i] + matchsticks[start]
			if num > svg {
				continue
			}
			arr[i] += matchsticks[start]
			if dfs(start + 1) {
				return true
			}
			arr[i] -= matchsticks[start]
		}
		return false
	}

	return dfs(0)
}

// 472. 连接词
func findAllConcatenatedWordsInADict(words []string) []string {
	m := make(map[string]bool)
	// 将每个字符串存储起来
	num := -1
	for _, val := range words {
		if len(val) != 0 {
			num = min(num, len(val))
		}
		m[val] = true
	}
	var dp func(str string) bool
	dp = func(str string) bool {
		slen := len(str)
		sdp := make([]bool, slen+1)
		// 通过两次循环 进行判断
		sdp[0] = true
		// 之所以等于 是因为 切片不包含最后的一位
		for i := 1; i <= slen; i++ {
			for j := 0; j < i; j++ {
				if !sdp[j] {
					continue
				}
				//	 判断区间存不存在，存在的话，就将结尾的部分变为true
				if m[str[j:i]] {
					sdp[i] = true
				}
			}
		}
		return sdp[slen]
	}
	res := []string{}
	for _, val := range words {
		if len(val) == 0 || len(val) < num*2 {
			continue
		}

		delete(m, val)
		if dp(val) {
			res = append(res, val)
		}
		m[val] = true
	}
	return res
}

// 470. 用 Rand7() 实现 Rand10()
func rand10() int {
	num := (rand7()-1)*7 + rand7()
	for num > 40 {
		num = (rand7()-1)*7 + rand7()
	}
	return 1 + num%10
}
func rand7() int {
	return 0
}

// 468. 验证IP地址
func validIPAddress(queryIP string) string {

	num := 0
	for _, val := range queryIP {
		if val == '.' {
			num = 4
		} else if val == ':' {
			num = 6
		}
	}
	split := []string{}
	if num == 4 {
		split = append(split, strings.Split(queryIP, ".")...)
	} else if num == 6 {
		split = append(split, strings.Split(queryIP, ":")...)
	} else {
		return "Neither"
	}

	if len(split) == 4 && isEmptyIpv4(split) {
		return "IPv4"
	}
	if len(split) == 8 && isEmptyIpv6(split) {
		return "IPv6"
	}
	return "Neither"
}

func isEmptyIpv6(arr []string) bool {

	for _, val := range arr {
		if len(val) == 0 {
			return false
		} else if len(val) > 4 {
			return false
		}
		for _, value := range val {
			if (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F') {
				continue
			} else {
				return false
			}
		}
	}

	return true
}

func isEmptyIpv4(arr []string) bool {
	for _, val := range arr {
		if len(val) == 0 {
			return false
		}
		if val[0] != '0' {
			atoi, _ := strconv.Atoi(val)
			if atoi >= 256 || atoi == 0 {
				return false
			}
		} else if len(val) > 1 {
			return false
		}
	}
	return true
}

// 467. 环绕字符串中唯一的子字符串
func findSubstringInWraproundString(p string) int {
	count := 0
	//	 计算每个唯一字符的个数
	dp := make([]int, 26)
	count_len := 1
	dp[p[0]-'a'] = 1
	for i := 1; i < len(p); i++ {
		if (p[i]-'a' == p[i-1]-'a'+1) || (p[i] == 'a' && p[i-1] == 'z') {
			count_len++
		} else {
			count_len = 1
		}
		dp[p[i]-'a'] = max(dp[p[i]-'a'], count_len)
	}
	for _, val := range dp {
		count += val
	}
	return count
}

//  466. 统计重复个数
func getMaxRepetitions(s1 string, n1 int, s2 string, n2 int) int {
	// 就是n1和s1组合中，能找到多少个s2，
	//	最后 找出来的个数,除以n2就可以了
	// 记录s2出现的次数 记录s2当前是哪个索引
	i, count, j := 0, 0, 0
	for i < n1 {
		for k := 0; k < len(s1); k++ {
			if s1[k] == s2[j] {
				// 如果一次s2走到头了之后 进行下一次的循环 次数+1
				if j == len(s2)-1 {
					count++
					j = 0
				} else {
					j++
				}
			}
		}
		i++
	}
	return count / n2
}

// 245. 最短单词距离 III
func shortestWordDistance(wordsDict []string, word1 string, word2 string) int {
	i, j := -1, -1
	num := math.MaxInt64
	if word1 == word2 {
		for key, val := range wordsDict {
			if val == word1 && i != j {
				num = min(num, numbers(key, i))
				i = key
			} else if val == word1 {
				i = key
			}
		}
	} else {
		for key, val := range wordsDict {
			if val == word1 {
				i = key
			} else if val == word2 {
				j = key
			}
			if i != -1 && j != -1 {
				num = min(num, numbers(i, j))
			}
		}
	}
	return num
}

//244. 最短单词距离 II
type WordDistance struct {
	strArr []string
}

func Constructor_244(wordsDict []string) WordDistance {
	return WordDistance{
		strArr: wordsDict,
	}
}

func (this *WordDistance) Shortest(word1 string, word2 string) int {
	if len(this.strArr) == 0 {
		return 0
	}
	i, j := -1, -1
	num := math.MaxInt64
	for key, val := range this.strArr {
		if val == word1 {
			i = key
		} else if val == word2 {
			j = key
		}
		if i != -1 && j != -1 {
			num = min(num, numbers(i, j))
		}
	}
	return num
}

// 243. 最短单词距离
func shortestDistance(wordsDict []string, word1 string, word2 string) int {
	i, j := -1, -1
	num := math.MaxInt64
	for key, val := range wordsDict {
		if val == word1 {
			i = key
		} else if val == word2 {
			j = key
		}
		if i != -1 && j != -1 {
			num = min(num, numbers(i, j))
		}
	}
	return num
}

// 186. 翻转字符串里的单词 II
func reverseWords(s []byte) {
	start, end := 0, len(s)-1
	for start < end {
		s[start], s[end] = s[end], s[start]
		start++
		end--
	}
	j := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			k := i - 1
			for j < k {
				s[j], s[k] = s[k], s[j]
				j++
				k--
			}
			j = i
		}
	}
	l := len(s) - 1
	for j < l {
		s[j], s[l] = s[l], s[j]
		j++
		l--
	}
}

// 464. 我能赢吗
func canIWin(maxChoosableInteger int, desiredTotal int) bool {
	// 当最大的选择数大于总和，直接返回true
	if maxChoosableInteger >= desiredTotal {
		return true
	}
	//	 当总和小于 规定的总和，那么都不可能完成返回false
	count := 0
	for i := 1; i <= maxChoosableInteger; i++ {
		count += i
	}
	if count < desiredTotal {
		return false
	}
	//	 采用记忆化的dfs key采用字符
	strArr := make([]byte, maxChoosableInteger+1)
	strMap := make(map[string]bool)
	var dfs func(total int) bool
	dfs = func(total int) bool {
		key := string(strArr)
		if val, ok := strMap[key]; ok {
			return val
		}
		for i := 1; i <= maxChoosableInteger; i++ {
			// 当前数字还没有被选中
			if strArr[i] == 0 {
				strArr[i] = 1
				// 要么当前选择之后满足条件，要么下面一个人选择不成功，才会返回true
				if total-i <= 0 || !dfs(total-i) {
					strMap[key] = true
					// 需要进行回溯
					strArr[i] = 0
					return true
				}
				// 上面赋值过后 这里进行回溯
				strArr[i] = 0
			}
		}
		strMap[key] = false
		return false
	}
	return dfs(desiredTotal)
}

// 463. 岛屿的周长
func islandPerimeter(grid [][]int) int {
	// 给外圈添加一个0
	n, m := len(grid), len(grid[0])
	row := make([]int, 1)
	for i := 0; i < n; i++ {
		grid[i] = append(row, grid[i]...)
		grid[i] = append(grid[i], row...)
	}
	arr := make([]int, m+2)
	grid = append([][]int{arr}, grid...)
	grid = append(grid, arr)
	var perimter func(i, j int) int
	perimter = func(i, j int) int {
		num := 4
		if grid[i-1][j] == 1 {
			num--
		}
		if grid[i+1][j] == 1 {
			num--
		}
		if grid[i][j-1] == 1 {
			num--
		}
		if grid[i][j+1] == 1 {
			num--
		}
		return num
	}
	count := 0
	for i := 1; i < len(grid)-1; i++ {
		for j := 1; j < len(grid[0])-1; j++ {
			if grid[i][j] == 1 {
				count += perimter(i, j)
			}

		}
	}
	return count
}

// 462. 最少移动次数使数组元素相等 II
func minMoves2(nums []int) int {
	sort.Ints(nums)
	num := len(nums) / 2
	count := 0
	for i := 0; i < len(nums); i++ {
		count += numbers(nums[num], nums[i])
	}
	return count
}

// LFUCache 460. LFU 缓存
type LFUCache struct {
	//	 cap 存放最大的容量 minFreq 最少使用频率
	cap, minFreq int
	//	通过一个map存放目前所有的k——v
	node map[int]*node
	// 通过一个map存放使用频率
	freqNode map[int]*doubleList
}

// 在nodemap中 表示存储的实体，在使用频率中 表示使用频率的双向链表
type node struct {
	// 新增的key ，val 以及调用的次数
	key, val, freq int
	prev, next     *node
}

func Constructor_460(capacity int) LFUCache {
	return LFUCache{
		cap:      capacity,
		node:     map[int]*node{},
		freqNode: map[int]*doubleList{},
		minFreq:  0,
	}
}

func (this *LFUCache) Get(key int) int {
	if val, ok := this.node[key]; ok {
		this.insertFreq(val)
		return val.val
	}
	return -1
}

func (this *LFUCache) Put(key int, value int) {
	//	 判断容量是否合法，
	//	判断key是否存在 存在就直接更新频率
	//	key不存在 新增，需要进行判断容量是否还有
	//	容量不足 删除使用频率最小的key
	//	容量足够 正常插入
	if this.cap <= 0 {
		return
	}
	if val, ok := this.node[key]; ok {
		val.val = value
		this.insertFreq(val)
		return
	}
	if len(this.node) >= this.cap {
		// 删除最小频率的key
		this.deleteFreq()
	}
	//	 新增key操作
	val := &node{
		key:  key,
		val:  value,
		freq: 1,
	}
	// 先将数据存储到所有的kv里面
	this.node[key] = val
	// 判断当前频率中是否有数据
	if this.freqNode[val.freq] == nil {
		// 当前频率没有 就初始化一个新的双向链表
		this.freqNode[val.freq] = newDoubleList()
	}
	// 双向链表新增数据
	this.freqNode[val.freq].add(val)
	this.minFreq = 1
}

// 删除频率最小的key
func (this *LFUCache) deleteFreq() {
	//	 获取频率最小的双向链表
	dl := this.freqNode[this.minFreq]
	// 获取双向链表的最后一个参数
	last := dl.last()
	//  删除双向链表最后的一个参数
	dl.remove(last)
	// 删除存储对应的kv
	delete(this.node, last.key)

}

// 增加node频率
func (this *LFUCache) insertFreq(node *node) {
	//	 先从doublelist中移除 然后查询对应的dl是否为空,查看是否是最低频率的链表
	//	是的话 移除过后查看是否为空，是的话，minfreq++ ，否则不变
	//	 node.freq++ 并加入到新的dl中
	freq := node.freq
	// 获取需要删除的数据的对应频率的双端链表数据
	oldL := this.freqNode[freq]
	oldL.remove(node)
	//	 判断删除过后双端链表是否为空
	if oldL.isEmpty() && freq == this.minFreq {
		this.minFreq++
	}
	// 将频率+1
	node.freq++
	// 判断新的这个频率里面有没有对应的数据
	if this.freqNode[node.freq] == nil {
		this.freqNode[node.freq] = newDoubleList()
	}
	this.freqNode[node.freq].add(node)

}

// 存储视频频率的双向链表
type doubleList struct {
	// 存放的是双端链表的头结点和尾节点
	head, tail *node
}

// 初始化双端链表
func newDoubleList() *doubleList {
	// 初始化的时候 节点都是自身绑定自身
	head, tail := &node{}, &node{}
	head.next, tail.prev = tail, head
	return &doubleList{
		head: head,
		tail: tail,
	}
}

// 新增双端链表
func (list *doubleList) add(node *node) {
	// 因为新增的是从头节点开始新增，所以先拿出虚拟的头结点和第一个节点
	prev, next := list.head, list.head.next
	// 新增进来的时候 新增的节点变成了第一个节点
	prev.next, node.prev = node, prev
	// 新增进来的时候，原来第一个节点变成了第二个节点
	next.prev, node.next = node, next
}

// 删除双向链表
func (list *doubleList) remove(node *node) {
	prev, next := node.prev, node.next
	//	将删除节点的上一个节点的下一个节点 指向删除节点的下一个节点
	//  将删除节点的下一个节点的上一个节点指向 删除节点的上一个节点
	prev.next, next.prev = next, prev
	//	将要删除的节点的上一个节点和下一个节点都置为空
	node.next, node.prev = nil, nil
}

// 获取头结点
func (list *doubleList) first() *node {
	return list.head.next
}

// 获取尾节点
func (list *doubleList) last() *node {
	return list.tail.prev
}

// 判断链表是否为空
func (list *doubleList) isEmpty() bool {
	return list.head.next == list.tail
}

// 459. 重复的子字符串
func repeatedSubstringPattern(s string) bool {
	slen := len(s)
	for i := 0; i < slen/2; i++ {
		num := i + 1
		if num != 1 && slen%num != 0 {
			continue
		}
		data := s[:i+1]
		j := 0
		flag := true
		for j < slen {
			if data != s[j:j+i+1] {
				flag = false
				break
			}
			j += i + 1
		}
		if flag {
			return flag
		}
	}

	return false
}

// 457. 环形数组是否存在循环
func circularArrayLoop(nums []int) bool {
	sum := len(nums)
	// 因为每一个元素都要去查找是否是正常的循环
	var dfs func(start int) bool
	dfs = func(start int) bool {
		cur := start
		// 判断当前是正数还是负数
		flag := nums[cur] > 0
		// key表示当前的步骤
		k := 1
		for {
			// 越界了 假如k进行了五次，但是数组长度只有四个，表示不满足
			if k > sum {
				return false
			}
			// 计算下一次寻走多少步，都换算为正数，如果是负数，加上当前的长度 然后取模
			next := ((cur+nums[cur])%sum + sum) % sum
			if flag && nums[next] <= 0 {
				return false
			}
			if !flag && nums[next] > 0 {
				return false
			}

			if next == start {
				return k > 1
			}
			cur = next
			k++
		}

	}

	for key, _ := range nums {
		if dfs(key) {
			return true
		}
	}
	return false
}

// 456. 132 模式
func find132pattern(nums []int) bool {
	// 采用单调栈进行操作 单调递减的栈进行操作
	// 这样就可以进行 i是最外层的循环，j是属于栈顶，k是属于栈底
	// 设置k的数
	k := math.MinInt64
	stack := []int{}
	for i := len(nums) - 1; i >= 0; i-- {
		//	 判断如果当前的栈不为空，栈里面是j，并且k大于i，那么就算符合条件
		if len(stack) != 0 && k > nums[i] {
			return true
		}
		//	 堆栈的数据进行计算 采用的是单调递增栈， 【6,5,4,3,2,1】
		//  判断数据是否可以进栈
		for len(stack) != 0 && nums[i] > stack[len(stack)-1] {
			// j是一个单调递增站，里面存储的是j，出栈的元素 寻找一个最大的值，k
			k = max(k, stack[len(stack)-1])
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, nums[i])
	}
	return false
}

// 455. 分发饼干
func findContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	count := 0
	glen, slen := len(g)-1, len(s)-1
	for glen >= 0 && slen >= 0 {
		if s[slen] >= g[glen] {
			slen--
			glen--
			count++
			continue
		}
		glen--
	}
	return count
}

// 454. 四数相加 II
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	m_1_2 := make(map[int]int)
	for _, val1 := range nums1 {
		for _, val2 := range nums2 {
			m_1_2[val1+val2]++
		}

	}
	count := 0
	for _, val3 := range nums3 {
		for _, val4 := range nums4 {
			count += m_1_2[-(val4 + val3)]

		}
	}
	return count
}

// 453. 最小操作次数使数组元素相等
func minMoves(nums []int) int {
	sort.Ints(nums)
	count := 0
	cur := nums[0]
	for _, val := range nums {
		count += val - cur
	}
	return count
}

// 452. 用最少数量的箭引爆气球
func findMinArrowShots(points [][]int) int {
	if len(points) == 0 {
		return 0
	}
	sort.Slice(points, func(i, j int) bool {
		return points[i][1] < points[j][1]
	})
	count := 1
	end := points[0][1]
	for i := 1; i < len(points); i++ {
		if points[i][0] > end {
			end = points[i][1]
			count++
		}
	}

	//fmt.Println(points)
	return count
}

// 451. 根据字符出现频率排序
func frequencySort(s string) string {
	by_451 := make([]Str_451, 75)
	for i := 0; i < len(by_451); i++ {
		by_451[i].str = string(rune(i + 48))
	}

	for i := 0; i < len(s); i++ {
		by_451[s[i]-48].by++
	}
	sort.Slice(by_451, func(i, j int) bool {
		return by_451[i].by > by_451[j].by
	})
	bytes := bytes.Buffer{}
	for _, val := range by_451 {
		if val.by == 0 {
			break
		}
		for i := 0; i < val.by; i++ {
			bytes.WriteString(val.str)
		}
	}
	return bytes.String()
}

type Str_451 struct {
	by  int
	str string
}

// 450. 删除二叉搜索树中的节点
func deleteNode(root *TreeNode, key int) *TreeNode {
	var dfs func(node *TreeNode) *TreeNode
	// 删除二叉树节点，
	dfs = func(node *TreeNode) *TreeNode {
		if node == nil {
			return nil
		}
		if node.Val == key {
			// 如果左子树为空，并且右子树为空 那么就直接返回nil
			if node.Left == nil && node.Right == nil {
				return nil
			}
			// 如果左子树为空，那么就直接将右子树置为当前节点
			if node.Left == nil {
				node = node.Right
				return node
			}
			// 如果右子树为空 那么就将左子节点成为当前节点返回
			if node.Right == nil {
				node = node.Left
				return node
			}
			// 如果两个节点都不为空，那么就有两种解题方式
			// 将当前的左子树插入右子树的最左子树
			// 或者将当前的右子树插入左子树的最右子树
			left := node.Left
			right := node.Right
			tmp := node.Left
			for tmp.Right != nil {
				tmp = tmp.Right
			}
			tmp.Right = right
			node = left
		}
		if key < node.Val {
			node.Left = dfs(node.Left)
		}
		if key > node.Val {
			node.Right = dfs(node.Right)
		}
		return node
	}
	return dfs(root)
}

// 449. 序列化和反序列化二叉搜索树
type Codec struct {
}

func Constructor_449() Codec {
	return Codec{}

}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	str := []string{}
	var dfs func(root *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			str = append(str, "-1")
			return
		}
		str = append(str, strconv.Itoa(node.Val))
		dfs(node.Left)
		dfs(node.Right)
	}
	dfs(root)
	return strings.Join(str, ",")
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	split := strings.Split(data, ",")
	var sub func() *TreeNode
	sub = func() *TreeNode {
		if len(split) == 0 {
			return nil
		}
		num, _ := strconv.Atoi(split[0])
		split = split[1:]
		if num < 0 {
			return nil
		}
		node := &TreeNode{}
		node.Val = num
		node.Left = sub()
		node.Right = sub()
		return node
	}
	return sub()
}

// 448. 找到所有数组中消失的数字
func findDisappearedNumbers(nums []int) []int {
	lenght := len(nums)
	m := make(map[int]bool)
	arr := []int{}
	for i := 0; i < lenght; i++ {
		m[nums[i]] = true
	}
	for i := 1; i <= lenght; i++ {
		if !m[i] {
			arr = append(arr, i)
		}
	}
	return arr
}

//447. 回旋镖的数量
func numberOfBoomerangs(points [][]int) int {
	var res int = 0
	m := make(map[int]int)
	for i := 0; i < len(points); i++ {
		m = make(map[int]int)
		for j := 0; j < len(points); j++ {
			if i == j {
				continue
			}
			distance := (points[i][0]-points[j][0])*(points[i][0]-points[j][0]) + (points[i][1]-points[j][1])*(points[i][1]-points[j][1])
			if _, ok := m[distance]; ok {
				res += m[distance] * 2
				m[distance]++
			} else {
				m[distance] = 1
			}
		}
	}
	return res
}

// 446. 等差数列划分 II - 子序列
func numberOfArithmeticSlices(nums []int) int {
	lenght := len(nums)
	if lenght < 3 {
		return 0
	}
	ans := 0
	dp := make([]map[int]int, lenght)
	for i := 0; i < lenght; i++ {
		// 初始化每个数的等差值map
		dp[i] = make(map[int]int)
		for j := 0; j < i; j++ {
			//	 获取i和j的等差数据
			num := nums[i] - nums[j]
			//	获取j为结尾，差值为num的等差数量
			cnt := dp[j][num]
			// 等差的数量等于，当前的值 加上原来的数，比如等差为2 发现  dp[j][num] 有两个等差为1的，那么这三个数就可以组成一个等差数列，
			// 如果为2 那么就是 i  j 还有两个数与i-j 与j-某两个数一样，表示两个等差数列，一次累加
			ans += cnt
			dp[i][num] += cnt + 1
		}
	}
	return ans
}

// 445. 两数相加 II
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	list1, _ := reverseList(l1)
	list2, _ := reverseList(l2)
	num := 0
	var node *ListNode
	for list1 != nil && list2 != nil {
		next := &ListNode{}
		sum := list1.Val + list2.Val + num
		if sum >= 10 {
			next.Val = sum % 10
			num = 1
		} else {
			next.Val = sum
			num = 0
		}
		next.Next = node
		node = next
		list1 = list1.Next
		list2 = list2.Next
	}
	for list1 != nil {
		next := &ListNode{}
		sum := list1.Val + num
		if sum >= 10 {
			num = 1
			next.Val = sum % 10
		} else {
			next.Val = sum
			num = 0
		}
		next.Next = node
		node = next
		list1 = list1.Next
	}
	for list2 != nil {
		next := &ListNode{}
		sum := list2.Val + num
		if sum >= 10 {
			num = 1
			next.Val = sum % 10
		} else {
			next.Val = sum
			num = 0
		}
		next.Next = node
		node = next
		list2 = list2.Next
	}
	if num == 1 {
		next := &ListNode{}
		next.Val = 1
		next.Next = node
		node = next
	}
	return node
}

// 反转链表
func reverseList(l1 *ListNode) (*ListNode, int) {
	var node *ListNode
	index := 0
	for l1 != nil {
		cur := l1.Next
		l1.Next = node
		node = l1
		l1 = cur
		index++
	}
	return node, index
}

// 445 采用数组方式实现
func arrAddTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	arr1, arr2 := []int{}, []int{}
	for l1 != nil {
		arr1 = append(arr1, l1.Val)
		l1 = l1.Next
	}
	for l2 != nil {
		arr2 = append(arr2, l2.Val)
		l2 = l2.Next
	}
	lenght := 0
	if len(arr2) > len(arr1) {
		arr1 = append(make([]int, len(arr2)-len(arr1)), arr1...)
		lenght = len(arr2)
	} else {
		arr2 = append(make([]int, len(arr1)-len(arr2)), arr2...)
		lenght = len(arr1)
	}
	var node *ListNode
	num := 0
	for i := lenght - 1; i >= 0; i-- {
		sum := arr1[i] + arr2[i] + num
		next := &ListNode{}
		if sum >= 10 {
			num = 1
			next.Val = sum % 10
		} else {
			num = 0
			next.Val = sum
		}
		next.Next = node
		node = next
	}
	if num == 1 {
		next := &ListNode{}
		next.Val = 1
		next.Next = node
		node = next
	}

	return node
}

// 443. 压缩字符串
func compress(chars []byte) int {
	start, end, res := 0, 0, 0
	for end < len(chars) {
		end++
		if end != len(chars) && chars[start] == chars[end] {
			continue
		}
		chars[res] = chars[start]
		res++
		num := end - start
		if num > 1 {
			str := strconv.Itoa(num)
			for i := 0; i < len(str); i++ {
				chars[res] = str[i]
				res++
			}
		}
		start = end
	}
	return res
}

// 442. 数组中重复的数据
func findDuplicates(nums []int) []int {
	arr := []int{}
	for i := 0; i < len(nums); i++ {
		if nums[abs(nums[i])-1] < 0 {
			arr = append(arr, abs(nums[i]))
		} else {
			nums[abs(nums[i])-1] *= -1
		}
	}
	return arr
}
func abs(i int) int {
	if i > 0 {
		return i
	}
	return -i
}

// 441. 排列硬币
func arrangeCoins(n int) int {
	if n == 0 {
		return 0
	}
	index := 0
	count := 0
	for {
		index++
		count++
		n -= index
		if n < 0 {
			break
		}
	}
	return count - 1
}
