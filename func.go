package func_master

import (
	md52 "crypto/md5"
	"encoding/hex"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

func Tag(cmd uint8, index uint16) int {
	return int(cmd)<<16 + int(index)
}

func Stop() {
	if !atomic.CompareAndSwapInt32(&stop, 0, 1) {
		return
	}
	close(StopChanForGo)
	for sc := 0; !WaitAll.TryWait(); sc++ {
		Sleep(1)
		if sc >= 3000 {
			LogError("Server Stop Timeout")
			stopCheckMap.Lock()
			for _, v := range stopCheckMap.M {
				LogError("Server Stop Timeout:%v", v)
			}
			stopCheckMap.Unlock()
			sc = 0
		}
	}
	LogInfo("Server Stop")
	close(StopChanForSys)
}

func IsRuning() bool {
	return stop == 0
}

func IsStop() bool {
	return stop == 1
}

// MD5加密
func MD5Str(s string) string {
	return MD5Bytes([]byte(s))
}

func MD5Bytes(s []byte) string {
	md5 := md52.New()
	md5.Write(s)
	cip := md5.Sum(nil)
	return hex.EncodeToString(cip)
}

func Md5File(path string) string {
	file, err := ReadFile(path)
	if err != nil {
		LogError("calc md5 failed path:%v", path)
		return ""
	}
	return MD5Bytes(file)
}

// WaitForSystemExit 等待系统退出
func WaitForSystemExit(atexit ...func()) {
	statis.StartTime = time.Now()
	signal.Notify(StopChanForSys, os.Interrupt, os.Kill, syscall.SIGTERM)

	select {
	case <-StopChanForSys:
		for _, v := range atexit {
			v()
		}
		Stop()
	}
	AtexitMapSync.Lock()
	for _, v := range atexitMap {
		v()
	}
	AtexitMapSync.Unlock()
	for _, v := range redisManagers {
		v.close()
	}

	WaitAllForRedis.Wait()
	if !atomic.CompareAndSwapInt32(&stopForLog, 0, 1) {
		return
	}
	WaitAll.Wait()
	close(StopChanForLog)
}

// Daemon 守护进程
func Daemon(skip ...string) {
	if os.Getppid() != 1 {
		abs, _ := filepath.Abs(os.Args[0])
		newCmd := []string{}
		for _, v := range os.Args {
			add := true
			for _, s := range skip {
				if strings.Contains(v, s) {
					add = false
					break
				}
			}
			if add {
				newCmd = append(newCmd, v)
			}
		}
		cmd := exec.Command(abs)
		cmd.Args = newCmd
		cmd.Start()
	}
}

func GetStatis() *Statis {
	statis.GoCount = int(gocount)
	statis.MsgqueCount = len(msgqueMap)
	return statis

}
