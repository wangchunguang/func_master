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
			id:            worker.GetId(),
			msgTyp:        msgtyp,
			handler:       handler,
			parserFactory: parser,
			connTyp:       ConnTypeListen,
		},
		address:  addr,
		listener: listener,
	}
	MsgqueMapSync.Lock()
	msgqueMap[msg.id] = &msg
	MsgqueMapSync.Unlock()
	return &msg
}

func newTcpConn(network, addr string, conn net.Conn, msgtyp MsgType, handler IMsgHandler, parser *Parser, user interface{}) *tcpMsgQue {
	msgque := tcpMsgQue{
		msgQue: msgQue{
			id:            worker.GetId(),
			cwrite:        make(chan *Message, 64),
			cread:         make(chan *Message, 64),
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
	MsgqueMapSync.Lock()
	msgqueMap[msgque.id] = &msgque
	MsgqueMapSync.Unlock()
	LogInfo("new msgque:%d remote addr:%s:%s", msgque.id, network, addr)
	return &msgque
}

func newTcpAccept(conn net.Conn, msgty MsgType, handler IMsgHandler, parser *Parser) *tcpMsgQue {
	msgque := tcpMsgQue{
		msgQue: msgQue{
			id:            worker.GetId(),
			cwrite:        make(chan *Message, 64),
			cread:         make(chan *Message, 64),
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
	MsgqueMapSync.Lock()
	msgqueMap[msgque.id] = &msgque
	MsgqueMapSync.Unlock()
	return &msgque
}

func newTcpGateWay(conn net.Conn, msgty MsgType, handler IMsgHandler, parser *Parser) *tcpMsgQue {
	msg := tcpMsgQue{
		msgQue: msgQue{
			id:            worker.GetId(),
			cwrite:        make(chan *Message, 64),
			cread:         make(chan *Message, 64),
			msgTyp:        msgty,
			handler:       handler,
			timeout:       DefMsgQueTimeout,
			connTyp:       ConnTypeGateWay,
			lastTick:      Timestamp,
			parserFactory: parser,
		},
		conn: conn,
	}

	if parser != nil {
		msg.parser = parser.Get()
	}
	MsgqueMapSync.Lock()
	msgqueMap[msg.id] = &msg
	MsgqueMapSync.Unlock()
	return &msg
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

// 读取客户端的数据进行处理
func (tcp *tcpMsgQue) readMsg() {
	// 消息头数据
	headData := make([]byte, MsgHeadSize)
	var data []byte
	var head *MessageHead
	for !tcp.IsStop() {
		err := tcp.conn.SetReadDeadline(time.Now().Add(someTimeout))
		if err != nil {
			LogError("Failed to read transfer timeout err :%s", err)
			break
		}
		if head == nil {
			_, err = io.ReadFull(tcp.conn, headData)
			if err != nil {
				if err != io.EOF {
					LogDebug("msgque:%v recv data err:%v", tcp.id, err)
				}
				break
			}
			if head = NewMessageHead(headData); head == nil {
				LogInfo("Did not get the data of the message.proto header headDate  :%s", headData)
				break
			}
			// 当客户端请求过来，没有网关分发的标识，直接返回
			if tcp.connTyp == ConnTypeGateWay && !head.Forward {
				LogInfo("Request the gateway not to forward")
				head = nil
				data = nil
				continue
			}
			if head.Len == 0 {
				if !tcp.processMsg(tcp, &Message{Head: head}) {
					LogError("msgque:%v process msg cmd:%v", tcp.id, head.Cmd)
					break
				}
				head = nil
			} else {
				data = make([]byte, head.Len)
			}
		} else {
			_, err = io.ReadFull(tcp.conn, data)
			if err != nil {
				LogError("msgque:%v recv data err:%v", tcp.id, err)
				break
			}
			if !tcp.processMsg(tcp, &Message{Head: head, Data: data}) {
				LogError("msgque:%v process msg cmd:%v ", tcp.id, head.Cmd)
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
	if tcp.connTyp == ConnTypeGateWay {
		tick := time.NewTimer(time.Second * time.Duration(tcp.timeout))
		for {
			select {
			// 将数据发布到指定的服务
			case msg := <-tcp.cread:
				tcp.gateWayWrite(msg)
			case <-tick.C:
				if tcp.isTimeout(tick) {
					break
				}
			}
		}
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
			// 当StopChanForGo 没有被关闭，或者100秒以内没有接受到数据，就会进行关闭
			select {
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

func (tcp *tcpMsgQue) gateWayWrite(msg *Message) {
	writeCount := 0
	hand := make([]byte, MsgHeadSize)
	// 判断是否有这个服务地址
	if value, ok := serverMap[ServerName]; ok && !tcp.IsStop() {
		addr := GateWayAddr(value)
		msg.Head.FastBytes(hand)
		write, err := addr.Coon.Write(hand[writeCount:])
		if err != nil {
			LogError("msgque write id:%v err:%v", tcp.id, err)
			goto WriteStop
		}
		writeCount += write
		if writeCount >= MsgHeadSize && msg.Data != nil {
			write, err = addr.Coon.Write(msg.Data[writeCount-MsgHeadSize : int(msg.Head.Len)])
			if err != nil {
				LogError("msgque write id:%v err:%v", tcp.id, err)
				goto WriteStop
			}
			writeCount += write
		}
		if writeCount == (int(msg.Head.Len) + MsgHeadSize) {
			writeCount = 0
		}
	}
WriteStop:
	Stop()
}

func (tcp *tcpMsgQue) gateway() {
	c := make(chan struct{})
	Go(func() {
		select {
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
		}
		msgque := newTcpGateWay(accept, tcp.msgTyp, tcp.handler, tcp.parserFactory)
		// 三次握手
		msgque.ShakeHands()
		if !tcp.interactive {
			break
		}
		msgque.SetEncrypt(tcp.GetEncrypt())
		// 表示是否有新的消息队列
		if tcp.handler.OnNewMsgQue(msgque) {
			msgque.init = true
			msgque.available = true
			Go(func() {
				tcp.readListen(msgque)
			})
			Go(func() {
				tcp.waiteListen(msgque)
			})
		} else {
			msgque.Stop()
		}
	}
	close(c)
	tcp.Stop()
}

func (tcp *tcpMsgQue) serverListen() {
	c := make(chan struct{})
	Go(func() {
		select {
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
		}
		msgque := newTcpAccept(accept, tcp.msgTyp, tcp.handler, tcp.parserFactory)
		msgque.SetEncrypt(tcp.GetEncrypt())
		if tcp.handler.OnNewMsgQue(msgque) {
			msgque.init = true
			msgque.available = true
			Go(func() {
				tcp.readListen(msgque)
			})
			Go(func() {
				tcp.waiteListen(msgque)
			})
		} else {
			msgque.Stop()
		}
	}
	close(c)
	tcp.Stop()

}

func (tcp *tcpMsgQue) readListen(msgque *tcpMsgQue) {
	LogInfo("process read for msgque:%d", msgque.id)
	msgque.read()
	LogInfo("process read end for msgque:%d", msgque.id)

}

func (tcp *tcpMsgQue) waiteListen(msgque *tcpMsgQue) {
	LogInfo("process read for msgque:%d", msgque.id)
	msgque.write()
	LogInfo("process read end for msgque:%d", msgque.id)

}

// ShakeHands 三次握手
func (tcp *tcpMsgQue) ShakeHands() {
	// 客户端加密种子数据
	iseedData := make([]byte, MsgClientSendSize)
	// 加密种子
	seedData := make([]byte, MsgSendSize)
	// 接收第三次握手客户端发送过来的加密种子
	data := make([]byte, MsgSendSize)
	// 第一次不用解密 因为不知道客户端的种子
	tcp.SeedRead(iseedData)
	tcp.SetSeed(iseedData)
	binary.BigEndian.PutUint32(seedData[:4], tcp.iseed)
	binary.BigEndian.PutUint32(seedData[4:], tcp.oseed)
	// 本次发送不需要加密，因为客户端还为获取服务器端的种子
	err := tcp.SeedWrite(seedData)
	if err != nil {
		LogError("Failed to receive client encrypted seed err :%s", err)
		Stop()
		return
	}
	tcp.SeedRead(data)
	// 解密判断数据是否正确
	decrypt := DefaultNetDecrypt(tcp.iseed, data, 0, MsgSendSize)
	iseed := binary.BigEndian.Uint32(decrypt[:4])
	oseed := binary.BigEndian.Uint32(decrypt[4:])
	if tcp.iseed != iseed || tcp.oseed != oseed {
		return
	}
	tcp.SetInteractive()
}

// SeedRead 读取客户端种子
func (tcp *tcpMsgQue) SeedRead(data []byte) {
	// 读取计数
	readCount := 0
	read, err := io.ReadFull(tcp.conn, data)
	if err != nil {
		LogError("Failed to set timeout err :%s", err)
		Stop()
		return
	}
	readCount += read
	if readCount < len(data) {
		_, err = io.ReadFull(tcp.conn, data[readCount:])
		if err != nil {
			Stop()
			return
		}
	}
}

// SeedWrite 写入加密种子到客户端
func (tcp *tcpMsgQue) SeedWrite(data []byte) error {
	// 写入计数
	writeCount := 0
	err := tcp.conn.SetWriteDeadline(time.Now().Add(TIMEOUT * time.Second))
	if err != nil {
		LogError("Failed to set timeout err :%s", err)
		return err
	}

	write, err := io.ReadFull(tcp.conn, data)
	if err != nil {
		LogError("Failed to write to the client err :%s", err)
		return err
	}
	writeCount += write
	if writeCount < len(data) {
		_, err = io.ReadFull(tcp.conn, data[writeCount:])
		LogError("Failed to write to the client err :%s", err)
		return err
	}
	return nil
}

// ClientConnect tcp客户端连接
func ClientConnect(addr string, network string) (net.Conn, bool) {
	c, err := net.DialTimeout(network, addr, time.Second)
	if err != nil {
		LogInfo("connect to addr:%s  err:%v", addr, err)
		return nil, false
	}
	return c, true
}
