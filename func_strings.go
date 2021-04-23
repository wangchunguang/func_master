package func_master

import (
	"errors"
	"strings"
)

func StrReplace(s, old, new string) string {
	return strings.Replace(s, old, new, -1)
}

const primeRK = 16777619

// strings.Builder用于写入数据相关操作
// strings.Reader用于读取数据相关操作  通过strings.NewReader
// strings.Replacer用于字符串替换，对多个goroutine使用是最安全的
// 获取字符串长度
func LenStr(str string) int {
	return len(str)
}

// 字符串比较，相等返回0，如果a小于b，返回-1，反之返回1
func Compare(a, b string) int {
	if a == b {
		return 0
	}
	if a < b {
		return -1
	}
	return +1
}

// 计算字符串中的hash值
func GetHash(sep string) uint32 {
	hash := uint32(0)
	for i := 0; i < len(sep); i++ {
		hash = hash*primeRK + uint32(sep[i])
	}
	return hash
}

// 计算字符逆序的hash值
func GetHashRev(sep string) uint32 {
	hash := uint32(0)
	for i := len(sep) - 1; i >= 0; i++ {
		hash = hash*primeRK + uint32(sep[i])
	}
	return hash
}

// 判断两个字符串忽略大小写之后是否相等
func StrEqualFold(a, b string) bool {
	fold := strings.EqualFold(a, b)
	return fold
}

// 判断是否存在某个字符或子串,子串str 在 s 中，返回 true
func StrContains(s, str string) bool {
	contains := strings.Contains(s, str)
	return contains
}

// 子串出现的次数,sep在s中出现的字数
func StrCount(s, sep string) int {
	count := strings.Count(s, sep)
	return count
}

// 通过空格截取字符串
func StrFields(s string) []string {
	fields := strings.Fields(s)
	return fields
}

// 切割字符串,不会保留切割的字符
func StrSplit(s, split string) []string {
	strs := strings.Split(s, split)
	return strs
}

// 切割字符串，保留切割的字符
func StrSplitAfter(s, split string) []string {
	after := strings.SplitAfter(s, split)
	return after
}

// 切割字符串，返回切片的个数
func StrSpiltN(s, split string, num int) []string {
	splitN := strings.SplitN(s, split, num)
	return splitN
}

// 判断字符串是否有某个前缀
func StrHasPrefix(str, pre string) bool {
	prefix := strings.HasPrefix(str, pre)
	return prefix
}

// 判断字符串是否有某个后缀
func StrHasSuffix(str, suf string) bool {
	suffix := strings.HasSuffix(str, suf)
	return suffix
}

// 获取字符或子串在字符串中出现的位置,在 sub 中查找 sep 的第一次出现，返回第一次出现的索引
func StrIndex(sep, sub string) int {
	n := len(sub)
	// 表示原始字符不存在
	if n == 0 {
		err := errors.New("The length of the searched character is 0")
		LogError(err)
		return -1
	}
	// 进行查找的字符不存在
	if len(sep) == 0 {
		err := errors.New("Character length is equal to 0")
		LogError(err)
	}
	switch {
	case n == 1:
		return strings.IndexByte(sep, sub[0])
	case n == len(sep):
		if sep == sub {
			return 0
		}
		return -1
	case n > len(sep):
		return -1
	case n < 64:
		// 测试运行时的CPU是否支持AVX2指令集
		// 可以的话 采用64进行判断
		if len(sep) <= 64 {
			// 短字符采用暴力破解算法
			return -1
		}
		c := sub[0]
		i := 0
		// 一段一段的获取
		t := sep[:len(sep)-n+1]
		fails := 0
		// 进行一个一个对比
		for i < len(t) {
			// 一个一个进行对比
			if t[i] != c {
				o := strings.IndexByte(t[i:], c)
				if o < 0 {
					return -1
				}
				i += o
			}
			if sep[i:i+n] == sub {
				return i
			}
			fails++
			i++
			if fails > (i+16)/8 {
				return -1
			}
		}

	}
	// 长字符采用RabinKarp 算法
	return -1

}

// 获取字节在字符串中第一次出现的索引
func StrIndexByte(s string, b byte) int {
	return strings.IndexByte(s, b)
}

// 获取字符最后一次出现的位置,在 s 中查找 sep 中最后一次出现的索引
func StrLastIndex(s, sep string) int {
	return strings.Index(s, sep)
}

// 获取字节最后一次出现在字符串中的位置
func StrLastIndexByte(s string, b byte) int {
	return strings.LastIndexByte(s, b)
}

// 连接字符串,通过sep连接字符串 fmt.Println(Join([]string{"name=xxx", "age=xx"}, "&")) 输出：name=xxx&age=xx
func StrJoin(s []string, sep string) string {
	return strings.Join(s, sep)
}

//  字符串重复几次 fmt.Println("ba" + strings.Repeat("na", 2)) 输出 banana
func StrRepeat(s string, count int) string {
	return strings.Repeat(s, count)
}

//  转换为小写字母
func StrToLower(s string) string {
	return strings.ToLower(s)
}

// 转换为大写
func StrToUpper(s string) string {
	return strings.ToUpper(s)
}

// 去除收尾空格
func StrTrimSpace(s string) string {
	return strings.TrimSpace(s)
}
