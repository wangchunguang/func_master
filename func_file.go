package func_master

import (
	"io"
	"io/ioutil"
	"os"
	"path"
	_ "path"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// 返回路径的最后一个元素
// 例 "/a/b" 输出 b
func PathBase(p string) string {
	return path.Base(p)
}

// 返回最短路径
// 例："/../a/b/../././/c" 输出 /a/c
func PathClean(p string) string {
	return path.Clean(p)
}

// 返回路径的目录
// 例 /a/b/c" 输出 /a/b
func PathDir(p string) string {
	return path.Dir(p)
}

// 返回文件的扩展名
// 例 /a/b/c/bar.css 输出.css
func PathExt(p string) string {
	return path.Ext(p)
}

// 返回文件的绝对路径
func PathAbs(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		LogError("get abs path failed path:%v err:%v", p, err)
		return ""
	}
	return abs
}

// 检查路径是否为绝对路径
func PathIsAbs(p string) bool {
	return path.IsAbs(p)
}

// 路径拼接
func PathJoin(p ...string) string {
	return path.Join(p...)
}

// 判断文件是否存在
func PathExists(p string) bool {
	_, err := os.Stat(p)
	if err == nil {
		return true
	}
	if !os.IsExist(err) {
		return false
	}
	return true
}

// 创建文件目录
func NewDir(p string) error {
	return os.MkdirAll(p, 0777)
}

// 读取文件
func ReadFile(p string) ([]byte, error) {
	file, err := ioutil.ReadFile(p)
	if err != nil {
		LogError("read file filed path:%v err:%v", p, err)
		return nil, ErrFileRead
	}
	return file, nil
}

// 写入文件
func WriteFile(p string, data []byte) {
	dir := PathDir(p)
	if !PathExists(p) {
		if err := NewDir(dir); err != nil {
			LogError("Failed to create file directory p：%v  err: %s", p, err)
			return
		}
	}
	ioutil.WriteFile(p, data, 0777)
}

// 获取文件下所有的文件名称
func GetFile(p string) []string {
	files := []string{}
	filepath.Walk(p, func(p string, f os.FileInfo, err error) error {
		if f == nil {
			return err
		}
		if f.IsDir() {
			return nil
		}
		files = append(files, p)
		return nil
	})
	return files
}

func DelFile(p string) {
	os.Remove(p)
}

func DelFileDir(p string) {
	os.RemoveAll(p)
}

func CreateFile(p string) (*os.File, error) {
	dir := PathDir(p)
	if PathExists(dir) {
		NewDir(dir)
	}
	return os.Create(p)
}

func CopyFile(dst io.Writer, src io.Reader) (written int64, err error) {
	return io.Copy(dst, src)
}

func walkDirTrue(dir string, wg *sync.WaitGroup, fun func(dir string, info os.FileInfo)) {
	wg.Add(1)
	defer wg.Done()
	infos, err := ioutil.ReadDir(dir)
	if err != nil {
		LogError("walk dir failed dir:%v err:%v", dir, err)
		return
	}
	for _, info := range infos {
		if info.IsDir() {
			fun(dir, info)
			subDir := filepath.Join(dir, info.Name())
			go walkDirTrue(subDir, wg, fun)
		} else {
			fun(dir, info)
		}
	}
}

func WalkDir(dir string, fun func(dir string, info os.FileInfo)) {
	if fun == nil {
		return
	}
	wg := &sync.WaitGroup{}
	walkDirTrue(dir, wg, fun)
	wg.Wait()
}

func FileCount(dir string) int32 {
	var count int32 = 0
	WalkDir(dir, func(dir string, info os.FileInfo) {
		if !info.IsDir() {
			atomic.AddInt32(&count, 1)
		}
	})
	return count
}

func DirCount(dir string) int32 {
	var count int32 = 0
	WalkDir(dir, func(dir string, info os.FileInfo) {
		if info.IsDir() {
			atomic.AddInt32(&count, 1)
		}
	})
	return count
}

func DirSize(dir string) int64 {
	var size int64 = 0
	WalkDir(dir, func(dir string, info os.FileInfo) {
		if !info.IsDir() {
			atomic.AddInt64(&size, info.Size())
		}
	})
	return size
}
