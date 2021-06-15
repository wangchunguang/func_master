package func_master

import (
	"time"
)

// 时间转换的特殊格式
const FORMAT = "2006-01-02 15:04:05"

func Date() string {
	return time.Now().Format(FORMAT)
}

func SetTimeout(inteval int, fn func(...interface{}) int, args ...interface{}) {
	if inteval < 0 {
		LogError("new timerout inteval:%v ms", inteval)
		return
	}
	LogInfo("new timerout inteval:%v ms", inteval)

	Go(func() {
		var tick *time.Timer
		for inteval > 0 {
			tick = time.NewTimer(time.Millisecond * time.Duration(inteval))
			select {
			case <-tick.C:
				tick.Stop()
				inteval = 0
			}
		}
	})
}

// 休眠 时间为微妙
func Sleep(num int) {
	time.Sleep(time.Microsecond * time.Duration(num))

}

// 获取当前时间
func GetTime() time.Time {
	now := time.Now()
	return now
}

// 获取当前年
func GetYear() int {
	year := GetTime().Year()
	return year
}

// 获取当前月
func GetMonth() int {
	month := GetTime().Month()
	return int(month)
}

// 获取当前日期
func GetDay() int {
	day := GetTime().Day()
	return day
}

// 获取当前是几点
func GetHour() int {
	hour := GetTime().Hour()
	return hour
}

// 获取当前时间的分钟数
func GetMinute() int {
	minute := GetTime().Minute()
	return minute
}

// 获取当前时间的时间搓
func GetUnix() int64 {
	unix := GetTime().Unix()
	return unix
}

// 只返回年月日
func GetDate() (year int, month time.Month, day int) {
	date, month, day := GetTime().Date()
	return date, month, day
}

// 计算时间差
func GetDuration(startTime, endTime time.Time) time.Duration {
	sub := endTime.Sub(startTime)
	return sub
}

// 计算相差的秒数
func SubSeconds(startTime, endTime time.Time) float64 {
	seconds := GetDuration(startTime, endTime).Seconds()
	return seconds
}

// 计算两个时间相差的纳秒数
func SubNanoseconds(startTime, endTime time.Time) int64 {
	nanoseconds := GetDuration(startTime, endTime).Nanoseconds()
	return nanoseconds
}

// 计算两个时间相差多少小时
func SubHour(startTime, endTime time.Time) float64 {
	hours := GetDuration(startTime, endTime).Hours()
	return hours
}

// 计算两个时间相差的分钟数
func SubMinute(startTime, endTime time.Time) float64 {
	minutes := GetDuration(startTime, endTime).Minutes()
	return minutes
}

// 暂停时间
func SetSleep(num int64) {
	time.Sleep(time.Second * time.Duration(num))
}

// 时间格式化  这个时间很特殊"2006-01-02 15:04:05" 必须是这个时间点的格式
func SetTimeToFormat(time time.Time, format string) string {
	timeFmt := time.Format(format)
	return timeFmt
}

// 将字符串解析成时间
func StringToTime(formatTimeStr string) time.Time {
	loc, _ := time.LoadLocation("Local")
	theTime, _ := time.ParseInLocation(FORMAT, formatTimeStr, loc)
	return theTime
}

// 将时间搓转换为字符串
func StampToString(stamp int64) string {
	format := time.Unix(stamp/1000.0, 0).Format(FORMAT)
	return format
}

// 将字符串解析为时间搓
func StringToStamp(formatTimeStr string) int64 {
	parsing := StringToTime(formatTimeStr)
	millisecond := parsing.UnixNano() / 1e6
	return millisecond
}

// 将时间转换为字符串
func TimeToString(t time.Time) string {
	temp := time.Date(t.YearDay(), t.Month(), t.Day(), t.Hour(), t.Minute(), t.Second(), t.Nanosecond(), time.Local)
	str := temp.Format(FORMAT)
	return str
}

func DateToUnix(date string) int64 {
	t, _ := time.ParseInLocation("2006-01-02 15:04:05", date, time.Local)
	return t.Unix()
}

// 将时间转换为时间搓 毫秒
func TimeToStamp(t time.Time) int64 {
	millisecond := t.UnixNano() / 1e6
	return millisecond
}

// 将时间搓转换为时间
func StampToTime(stamp int64) time.Time {
	stampStr := StampToString(stamp)
	timer := StringToTime(stampStr)
	return timer
}

func timerTick() {
	StartTick = time.Now().UnixNano() / 1000000
	NowTick = StartTick
	Timestamp = NowTick / 1000
	Go(func() {
		for IsRuning() {
			Sleep(1)
			NowTick = time.Now().UnixNano() / 1000000
			Timestamp = NowTick / 1000
		}
	})
}
