package func_master

import (
	"encoding/binary"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"
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

// GetNetType 获取tcp类型
func (tcp *tcpMsgQue) GetNetType() NetType {
	return NetTypeTcp

}

// Stop 停止
func (tcp *tcpMsgQue) Stop() {
	// 当stop为0 的时候
	if atomic.CompareAndSwapInt32(&tcp.stop, 0, 1) {
		Go(func() {
			if tcp.init {
				tcp.handler.OnDelMsgQue(tcp)
				if tcp.connecting == 1 {
					tcp.available = false
					return
				}
			}
			tcp.available = false
			tcp.baseStop()
			if tcp.conn != nil {
				tcp.conn.Close()
			}
			if tcp.listener != nil {
				tcp.listener.Close()
			}
		})
	}
}

func (tcp *tcpMsgQue) IsStop() bool {
	if tcp.stop == 0 {
		if IsStop() {
			tcp.Stop()
		}
	}
	return tcp.stop == 1
}

// LocalAddr 获取网络地址
func (tcp *tcpMsgQue) LocalAddr() string {
	if tcp.listener != nil {
		return tcp.listener.Addr().String()
	} else if tcp.conn != nil {
		return tcp.conn.LocalAddr().String()
	}
	return ""
}

// RemoteAddr 删除网络地址
func (tcp *tcpMsgQue) RemoteAddr() string {
	if tcp.conn != nil {
		return tcp.conn.RemoteAddr().String()
	}
	return ""

}

func (tcp *tcpMsgQue) Connect() bool {
	LogInfo("connect to addr:%s msgque:%d start", tcp.address, tcp.id)
	c, err := net.DialTimeout(tcp.network, tcp.address, time.Second)
	if err != nil {
		LogInfo("connect to addr:%s msgque:%d err:%v", tcp.address, tcp.id, err)
		tcp.handler.OnConnectComplete(tcp, false)
		atomic.CompareAndSwapInt32(&tcp.connecting, 1, 0)
		tcp.Stop()
		return false
	} else {
		tcp.conn = c
		tcp.available = true
		LogInfo("connect to addr:%s msgque:%d sucess", tcp.address, tcp.id)
		if tcp.handler.OnConnectComplete(tcp, true) {
			atomic.CompareAndSwapInt32(&tcp.connecting, 1, 0)
			Go(func() { tcp.read() })
			Go(func() { tcp.write() })
			return true
		} else {
			atomic.CompareAndSwapInt32(&tcp.connecting, 1, 0)
			tcp.Stop()
			return false
		}
	}
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

func newTcpConn(network, addr string, conn net.Conn, msgtyp MsgType, handler IMsgHandler, parser *Parser, user interface{}) *tcpMsgQue {
	msgque := tcpMsgQue{
		msgQue: msgQue{
			id:            atomic.AddUint32(&msgqueId, 1),
			cwrite:        make(chan *Message, 64),
			msgTyp:        msgtyp,
			handler:       handler,
			timeout:       DefMsgQueTimeout,
			connTyp:       ConnTypeConn,
			parserFactory: parser,
			lastTick:      Timestamp,
			user:          user,
		},
		conn:    conn,
		network: network,
		address: addr,
	}
	if parser != nil {
		msgque.parser = parser.Get()
	}
	msgqueMapSync.Lock()
	msgqueMap[msgque.id] = &msgque
	msgqueMapSync.Unlock()
	LogInfo("new msgque:%d remote addr:%s:%s", msgque.id, network, addr)
	return &msgque
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

func (tcp *tcpMsgQue) read() {
	tcp.wait.Add(1)
	defer func() {
		tcp.wait.Done()
		if err := recover(); err != nil {
			LogError("msgque read panic id:%v err:%v", tcp.id, err.(error))
			LogStack()
		}
		tcp.Stop()
	}()
	tcp.readMsg()
}

// SetEncrypt 设置是否加密
func (mq *msgQue) SetEncrypt(e bool) {
	mq.encrypt = e
}

// EncryptedServerSeed 客户端与服务器的第一次通信，将加密种子传给客户端
func (tcp *tcpMsgQue) EncryptedServerSeed() {
	if tcp.connTyp == ConnTypeAccept {
		tcp.oseed = uint32(Timestamp)
		data := make([]byte, 4)
		binary.BigEndian.PutUint32(data, tcp.oseed)
		// 第一次握手
		msg := NewMsg(uint8(HandCmd), uint8(HandAct), 0, 0, data)
		//  将加密种子传给客户端
		tcp.cwrite <- msg
	}
}

// SetIseed 设置加密种子
func (tcp *tcpMsgQue) SetIseed(msg *Message) bool {
	// 作为客户端时，内部服务器调用
	if tcp.connTyp == ConnTypeConn && msg.Head.Cmd == 0 {
		tcp.encrypt = true
		tcp.oseed = binary.BigEndian.Uint32(msg.Data[:4])
		tcp.iseed = binary.BigEndian.Uint32(msg.Data[4:])
		return true
	}
	// 数据经过加密的 用于三次握手  传输种子需要加密
	if tcp.encrypt && msg.Head != nil && msg.Head.Flags&FlagEncrypt > 0 && msg.Head.Cmd == uint8(HandCmd) {
		msg.Data = DefaultNetDecrypt(tcp.iseed, msg.Data, 0, msg.Head.Len)
		return tcp.SendCwriteData(msg)
	}
	return false
}

func (tcp *tcpMsgQue) SendCwriteData(msg *Message) bool {
	data := make([]byte, 8)
	tcp.iseed = binary.BigEndian.Uint32(msg.Data)
	binary.BigEndian.PutUint32(data, tcp.iseed)
	binary.BigEndian.PutUint32(data[4:], tcp.oseed)
	m := NewMsg(0, 1, 0, 0, data)
	tcp.cwrite <- m
	return true
}

// 读取客户端的数据进行处理
func (tcp *tcpMsgQue) readMsg() {
	// 消息头数据
	headData := make([]byte, MsgHeadSize)
	var data []byte
	var head *MessageHead
	for !tcp.IsStop() {
		if head == nil {
			_, err := io.ReadFull(tcp.conn, headData)
			if err != nil {
				if err != io.EOF {
					LogDebug("msgque:%v recv data err:%v", tcp.id, err)
				}
				break
			}
			if head = NewMessageHead(headData); head == nil {
				LogInfo("Did not get the data of the message header headDate  :%s", headData)
				break
			}
			if head.Len == 0 {
				if !tcp.processMsg(tcp, &Message{Head: head}) {
					LogError("msgque:%v process msg cmd:%v act:%v", tcp.id, head.Cmd, head.Act)
					break
				}
				head = nil
			} else {
				data = make([]byte, head.Len)
			}
		} else {
			_, err := io.ReadFull(tcp.conn, data)
			if err != nil {
				LogError("msgque:%v recv data err:%v", tcp.id, err)
				break
			}
			msg := &Message{Head: head, Data: data}
			// 作为客户端时内部调用或者三次握手的时候进行调用
			if tcp.SetIseed(msg) {
				head = nil
				data = nil
				continue
			}
			if !tcp.processMsg(tcp, msg) {
				LogError("msgque:%v process msg cmd:%v act:%v", tcp.id, head.Cmd, head.Act)
				break
			}
			head = nil
			data = nil
		}
		tcp.lastTick = Timestamp
	}
}

// 将数据传输到客户端
func (tcp *tcpMsgQue) write() {
	defer func() {
		tcp.wait.Done()
		if err := recover(); err != nil {
			LogError("msgque write panic id:%v err:%v", tcp.id, err.(error))
			LogStack()
		}
		tcp.Stop()
	}()
	tcp.wait.Add(1)
	// 是否快速发送
	if tcp.sendFast {
		//没有消息头
		tcp.writeMsgFast()
	} else {
		tcp.writeMsg()
	}

}

func (tcp *tcpMsgQue) writeMsgFast() {
	var m *Message
	var data []byte
	writeCount := 0
	tick := time.NewTimer(time.Second * time.Duration(tcp.timeout))
	for !tcp.IsStop() || m != nil {
		if m == nil {
			// 当stopChanForGo 没有被关闭，或者100秒以内没有接受到数据，就会进行关闭
			select {
			case <-stopChanForGo:
			case m = <-tcp.cwrite:
				if m != nil {
					data = m.Data
				}
			case <-tick.C:
				if tcp.isTimeout(tick) {
					tcp.Stop()
				}
			}
		}
		if m == nil {
			continue
		}
		if writeCount < len(data) {
			write, err := tcp.conn.Write(data[writeCount:])
			if err != nil {
				LogError("msgque write id:%v err:%v", tcp.id, err)
				break
			}
			writeCount += write
		}
		if writeCount == len(data) {
			writeCount = 0
			m = nil
		}
		tcp.lastTick = Timestamp
	}
	tick.Stop()
}

func (tcp *tcpMsgQue) writeMsg() {
	var m *Message
	hand := make([]byte, MsgHeadSize)
	writeCount := 0
	tick := time.NewTimer(time.Second * time.Duration(tcp.timeout))
	for !IsStop() || m != nil {
		if m == nil {
			select {
			case <-stopChanForGo:
			case m = <-tcp.cwrite:
				if m != nil {

					if tcp.encrypt && m.Head != nil && m.Head.Cmd != 0 {
						m = m.Copy()
						m.Head.Flags |= FlagEncrypt
						m.Head.Bcc = CountBCC(m.Data, 0, m.Head.Len)
						m.Data = DefaultNetEncrypt(tcp.oseed, m.Data, 0, m.Head.Len)
					}
					m.Head.FastBytes(hand)
				}
			case <-tick.C:
				if tcp.isTimeout(tick) {
					break
				}
			}
		}
		if m == nil {
			continue
		}
		if writeCount < MsgHeadSize {
			write, err := tcp.conn.Write(hand[writeCount:])
			if err != nil {
				LogError("msgque write id:%v err:%v", tcp.id, err)
				break
			}
			writeCount += write
		}
		if writeCount >= MsgHeadSize && m.Data != nil {
			write, err := tcp.conn.Write(m.Data[writeCount-MsgHeadSize : int(m.Head.Len)])
			if err != nil {
				LogError("msgque write id:%v err:%v", tcp.id, err)
				break
			}
			writeCount += write
		}
		if writeCount == int(m.Head.Len)+MsgHeadSize {
			writeCount = 0
			m = nil
		}
	}
	tcp.Stop()

}

func (tcp *tcpMsgQue) listen() {
	c := make(chan struct{})
	Go2(func(cstop chan struct{}) {
		select {
		case <-cstop:
		case <-c:
		}
		tcp.listener.Close()
	})
	for !tcp.IsStop() {
		accept, err := tcp.listener.Accept()
		if err != nil {
			if stop == 0 && tcp.stop == 0 {
				LogInfo("Message acceptance failed msgque :%v  err :%s", tcp.id, err)
			}
			break
		} else {
			Go(func() {
				// 初始化tcp接收
				msgque := newTcpAccept(accept, tcp.msgTyp, tcp.handler, tcp.parserFactory)
				msgque.SetEncrypt(tcp.GetEncrypt())
				// 向客户端发送服务器的种子
				msgque.EncryptedServerSeed()
				if tcp.handler.OnNewMsgQue(msgque) {
					msgque.init = true
					msgque.available = true
					Go(func() {
						LogInfo("process read for msgque:%d", msgque.id)
						msgque.read()
						LogInfo("process read end for msgque:%d", msgque.id)
					})
					Go(func() {
						LogInfo("process read for msgque:%d", msgque.id)
						msgque.write()
						LogInfo("process read end for msgque:%d", msgque.id)
					})
				} else {
					msgque.Stop()
				}
			})
		}
	}
	close(c)
	tcp.Stop()
}
