package func_master

import (
	"strconv"
)

func Atoi(str string) int {
	i, err := strconv.Atoi(str)
	if err != nil {
		return 0
	}
	return i
}

func Atoi32(str string) int32 {
	return int32(Atoi(str))
}

func Atoi64(str string) int64 {
	parseInt, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		LogError("String conversion failed str :%v  err :%s", str, err.Error())
		return 0
	}
	return parseInt
}

func Atof(str string) float32 {
	float, err := strconv.ParseFloat(str, 32)
	if err != nil {
		LogError("Conversion failed float32 str :%v err :%s", str, err.Error())
		return 0
	}
	return float32(float)
}

func Atof64(str string) float64 {
	float, err := strconv.ParseFloat(str, 64)
	if err != nil {
		LogError("Conversion failed float64 str :%v err :%s", str, err.Error())
		return 0
	}
	return float
}

func Itoa(num interface{}) string {
	switch n := num.(type) {
	case int8:
		return strconv.FormatInt(int64(n), 10)
	case int16:
		return strconv.FormatInt(int64(n), 10)
	case int32:
		return strconv.FormatInt(int64(n), 10)
	case int64:
		return strconv.FormatInt(n, 10)
	case uint8:
		return strconv.FormatUint(uint64(n), 10)
	case uint16:
		return strconv.FormatUint(uint64(n), 10)
	case uint32:
		return strconv.FormatUint(uint64(n), 10)
	case uint64:
		return strconv.FormatUint(n, 10)
	}
	return ""
}
