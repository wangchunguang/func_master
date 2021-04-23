package func_master

import (
	"net"
	"sync"
	"sync/atomic"
)

type tcpMsgQue struct {
	msgQue
	conn       net.Conn     // tcp连接
	listener   net.Listener // 监听
	network    string       // 网络 这里指的是tcp
	address    string       // 地址
	wait       sync.WaitGroup
	connecting int32 // 为true表示连接中
}


// 获取tcp类型
func (t *tcpMsgQue) GetNetType() NetType {
	return NetTypeTcp

}

// 停止
func (t *tcpMsgQue) Stop() {
	// 当stop为0 的时候
	if atomic.CompareAndSwapInt32(&t.stop, 0, 1) {
		Go(func() {
			if t.init {
				t.handler.OnDelMsgQue(t)
				if t.connecting == 1 {
					t.available = false
					return
				}
			}
			t.available = false
			t.baseStop()
			if t.conn != nil {
				t.conn.Close()
			}
			if t.listener != nil {
				t.listener.Close()
			}
		})
	}
}

func (t *tcpMsgQue) IsStop() bool {
	if t.stop == 0 {
		if IsStop() {
			t.Stop()
		}
	}
	return t.stop == 1
}

// 获取网络地址
func (t *tcpMsgQue) LocalAddr() string {
	if t.listener != nil {
		return t.listener.Addr().String()
	} else if t.conn != nil {
		return t.conn.LocalAddr().String()
	}
	return ""
}


// 删除网络地址
func (t *tcpMsgQue) RemoteAddr() string {
	if t.conn != nil {
		return t.conn.RemoteAddr().String()
	}
	return ""

}
