package func_master

import (
	"io"
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

func NewTcpListen(listener net.Listener, msgtyp MsgType, handler IMsgHandler, parser *Parser, addr string) *tcpMsgQue {
	msg := tcpMsgQue{
		msgQue: msgQue{
			id:            atomic.AddUint32(&msgqueId, 1),
			msgTyp:        msgtyp,
			handler:       handler,
			parserFactory: parser,
			connTyp:       ConnTypeListen,
		},
		listener: listener,
	}
	msgqueMapSync.Lock()
	msgqueMap[msg.id] = &msg
	msgqueMapSync.Unlock()
	return &msg
}

func newTcpAccept(conn net.Conn, msgty MsgType, handler IMsgHandler, parser *Parser) *tcpMsgQue {
	msgque := tcpMsgQue{
		msgQue: msgQue{
			id:            atomic.AddUint32(&msgqueId, 1),
			cwrite:        make(chan *Message, 64),
			msgTyp:        msgty,
			handler:       handler,
			timeout:       DefMsgQueTimeout,
			connTyp:       ConnTypeAccept,
			lastTick:      Timestamp,
			parserFactory: parser,
		},
		conn: conn,
	}

	if parser != nil {
		msgque.parser = parser.Get()
	}
	tcpConn, ok := conn.(*net.TCPConn)
	if !ok {
		LogError("TCPConn error")
	}
	tcpConn.SetNoDelay(true)
	msgqueMapSync.Lock()
	msgqueMap[msgque.id] = &msgque
	msgqueMapSync.Unlock()
	return &msgque
}

func (r *tcpMsgQue) read() {
	r.wait.Add(1)
	defer func() {
		r.wait.Done()
		if err := recover(); err != nil {
			LogError("msgque read panic id:%v err:%v", r.id, err.(error))
			LogStack()
		}
		r.Stop()
	}()
	r.readMsg()
}

func (r *tcpMsgQue) readMsg() {
	// 设置信息头大小
	headData := make([]byte, MsgHeadSize)
	var data []byte
	var head *MessageHead
	for !r.IsStop() {
		if head == nil {
			_, err := io.ReadFull(r.conn, headData)
			if err != nil {
				if err != io.EOF {
					LogDebug("msgque:%v recv data err:%v", r.id, err)
				}
				break
			}
			if head = NewMessageHead(headData); head == nil {
				LogInfo("Did not get the data of the message header headDate  :%s", headData)
				break
			}
		} else {

		}

	}
}

func (r *tcpMsgQue) listen() {
	c := make(chan struct{})
	Go2(func(cstop chan struct{}) {
		select {
		case <-cstop:
		case <-c:
		}
		r.listener.Close()
	})
	for !r.IsStop() {
		accept, err := r.listener.Accept()
		if err != nil {
			if stop == 0 && r.stop == 0 {
				LogInfo("Message acceptance failed msgque :%v  err :%s", r.id, err)
			}
			break
		} else {
			Go(func() {
				// 初始化tcp接收
				msgque := newTcpAccept(accept, r.msgTyp, r.handler, r.parserFactory)
				// 设置是否加密
				msgque.SetEncrypt(r.GetEncrypt())
				if r.handler.OnNewMsgQue(msgque) {
					msgque.init = true
					msgque.available = true
					Go(func() {
						LogInfo("process read for msgque:%d", msgque.id)
						msgque.read()
						LogInfo("process read end for msgque:%d", msgque.id)
					})
				} else {
					msgque.Stop()
				}
			})
		}
	}
	close(c)
	r.Stop()
}
