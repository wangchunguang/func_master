package func_master

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io/ioutil"
	"reflect"
	"sync"
	"time"
)

var DefMsgQueTimeout int = 180

// 发送的长度
var cwrite_chan_len = 64
var cread_chan_len = 64

type MsgType int

const (
	MsgTypeMsg MsgType = iota //消息基于确定的消息头
)

type NetType int

const (
	NetTypeTcp NetType = iota //TCP类型
)

type ConnType int

const (
	ConnTypeListen ConnType = iota //监听
	ConnTypeConn                   //连接产生的
	ConnTypeAccept                 //Accept产生的
)

type IMsgQue interface {
	Id() uint32
	GetMsgType() MsgType
	GetConnType() ConnType
	GetNetType() NetType

	LocalAddr() string
	RemoteAddr() string

	Stop()
	IsStop() bool
	Available() bool

	Send(m *Message) (re bool)
	SendString(str string) (re bool)
	SendStringLn(str string) (re bool)
	SendByteStr(str []byte) (re bool)
	SendByteStrLn(str []byte) (re bool)
	SendCallback(m *Message, c chan *Message) (re bool)
	SetSendFast()
	SetTimeout(t int)
	GetTimeout() int
	Reconnect(t int) //重连间隔  最小1s，此函数仅能连接关闭是调用

	GetHandler() IMsgHandler

	SetUser(user interface{})
	GetUser() interface{}

	Store(key, value interface{})
	Load(key interface{}) (value interface{}, ok bool)
	Delete(key interface{})
	tryCallback(msg *Message) (re bool)
}

type msgQue struct {
	sync.Map
	id            uint32        //唯一标示
	cwrite        chan *Message //写入通道
	stop          int32         //停止标记
	msgTyp        MsgType       //消息类型
	connTyp       ConnType      //通道类型
	handler       IMsgHandler   //处理者
	parser        IParser       // 解析器相关接口
	parserFactory *Parser       // 解析器的工程类
	timeout       int           //传输超时
	lastTick      int64
	init          bool
	available     bool
	sendFast      bool
	callback      map[int]chan *Message
	user          interface{}
	callbackLock  sync.Mutex
	encrypt       bool   //通信是否加密
	iseed         uint32 //input种子
	oseed         uint32 //output种子
}

// 设置是否快速发送
func (r *msgQue) SetSendFast() {
	r.sendFast = true
}

// 返回通道的id
func (r *msgQue) Id() uint32 {
	return r.id
}

// 消息头，
func (r *msgQue) GetMsgType() MsgType {
	return r.msgTyp
}

// 返回通道类型
func (r *msgQue) GetConnType() ConnType {
	return r.connTyp
}

// 是否停止
func (r *msgQue) Available() bool {
	return r.available
}

// 发送
func (r *msgQue) Send(m *Message) (re bool) {
	if m == nil || r.Available() {
		return
	}
	// 如果写入里面的长度 大于设置的长度，直接返回
	if len(r.cwrite) > cwrite_chan_len-1 {
		return
	}
	defer func() {
		if err := recover(); err != nil {
			re = false
		}
	}()
	r.cwrite <- m
	return true
}

// 向里面添加数据
func (r *msgQue) SendString(str string) (re bool) {
	return r.Send(&Message{Data: []byte(str)})
}

func (r *msgQue) SendStringLn(str string) (re bool) {
	return r.SendString(str + "\n")

}

func (r *msgQue) SendByteStr(str []byte) (re bool) {
	return r.SendString(string(str))
}

func (r *msgQue) SendByteStrLn(str []byte) (re bool) {
	return r.SendString(string(str) + "\n")
}

func (r *msgQue) SendCallback(m *Message, c chan *Message) (re bool) {
	if c == nil || cap(c) < 1 {
		LogError("try send callback but chan is null or no buffer")
		return
	}
	if r.Send(m) {
		r.setCallback(m.Tag(), c)
	} else {
		c <- nil
		return
	}
	return true
}

// 设置回调
func (r *msgQue) setCallback(tag int, c chan *Message) {
	r.callbackLock.Lock()
	defer func() {
		if err := recover(); err != nil {
			LogError("msgQue setCallback failure")
		}
		r.callback[tag] = c
		r.callbackLock.Unlock()
	}()
	if r.callback == nil {
		r.callback = make(map[int]chan *Message)
	}
	oc, ok := r.callback[tag]
	if ok {
		oc <- nil
	}

}

// 设置超时
func (r *msgQue) SetTimeout(t int) {
	if t > 0 {
		r.timeout = t
	}
}

// 查询超时
func (r *msgQue) GetTimeout() int {
	return r.timeout
}

// 是否超时
func (r *msgQue) isTimeout(tick *time.Timer) bool {
	left := int(Timestamp - r.lastTick)
	if left < r.timeout || r.timeout == 0 {
		if r.timeout == 0 {
			tick.Reset(time.Second * time.Duration(DefMsgQueTimeout))
		} else {
			tick.Reset(time.Second * time.Duration(r.timeout-left))
		}
		return false
	}
	LogInfo("msgque close because timeout id:%v wait:%v timeout:%v", r.id, left, r.timeout)
	return true
}

func (r *msgQue) Reconnect(t int) {

}

// 处理消息程序
func (r *msgQue) GetHandler() IMsgHandler {
	return r.handler
}

// 新增用户
func (r *msgQue) SetUser(user interface{}) {
	r.user = user
}

// 获取用户
func (r *msgQue) GetUser() interface{} {
	return r.user
}

// 尝试回拨
func (r *msgQue) tryCallback(msg *Message) (re bool) {
	if r.callback == nil {
		return false
	}
	defer func() {
		if err := recover(); err != nil {

		}
		r.callbackLock.Unlock()
	}()
	r.callbackLock.Lock()
	if r.callback != nil {
		tag := msg.Tag()
		if c, ok := r.callback[tag]; ok {
			delete(r.callback, tag)
			c <- msg
			re = true
		}
	}
	return
}

func (r *msgQue) baseStop() {
	if r.cwrite != nil {
		close(r.cwrite)
	}
	for k, v := range r.callback {
		v <- nil
		delete(r.callback, k)
	}
	msgqueMapSync.Lock()
	delete(msgqueMap, r.id)
	msgqueMapSync.Unlock()
}

// 处理消息
func (r *msgQue) processMsg(msgque IMsgQue, msg *Message) bool {
	if r.connTyp == ConnTypeConn && msg.Head.Cmd == 0 && msg.Head.Act == 0 {
		if len(msg.Data) != 8 {
			LogWarn("init seed msg err")
			return false
		}
		r.encrypt = true
		r.oseed = binary.BigEndian.Uint32(msg.Data[:4])
		r.iseed = binary.BigEndian.Uint32(msg.Data[4:])
		return true
	}
	// 数据经过加密的
	if msg.Head != nil && msg.Head.Flags&FlagEncrypt > 0 {
		r.iseed = r.iseed*cryptA + cryptB
		msg.Data = DefaultNetDecrypt(r.iseed, msg.Data, 0, msg.Head.Len)
		bcc := CountBCC(msg.Data, 0, msg.Head.Len)
		if msg.Head.Bcc != bcc {
			LogWarn("client bcc err conn:%d", r.id)
			return false
		}
	}
	// 数据经过压缩的
	if msg.Head != nil && msg.Head.Flags&FlagCompress > 0 && msg.Data != nil {
		data, err := GZipUnCompress(msg.Data)
		if err != nil {
			LogError("msgque uncompress failed msgque:%v cmd:%v act:%v len:%v err:%v", msgque.Id(), msg.Head.Cmd, msg.Head.Act, msg.Head.Len, err)
			return false
		}
		msg.Data = data
		msg.Head.Len = uint32(len(msg.Data))
	}
	if r.parser != nil {
		mp, err := r.parser.ParseC2S(msg)

		if err == nil {
			msg.IMsgParser = mp
		} else {
			if r.parser.GetErrType() == ParseErrTypeSendRemind {
				//机器人作为客户端时，不能全部注册消息，但不能断开连接
				if r.connTyp == ConnTypeConn {
					return true
				}

				return false
			} else if r.parser.GetErrType() == ParseErrTypeClose {
				return false
			} else if r.parser.GetErrType() == ParseErrTypeContinue {
				return true
			}
		}
	}
	f := r.handler.GetHandlerFunc(msgque, msg)
	if f == nil {
		f = r.handler.OnProcessMsg
	}
	return f(msgque, msg)

}

// 解密
// seed 加密解密种子，  buf 数据，开始的下标位置 长度
func DefaultNetDecrypt(seed uint32, buf []byte, offset uint32, len uint32) []byte {
	if len < offset {
		LogError("Decryption length is not enough")
		return buf
	}
	b_buf := bytes.NewBuffer([]byte{})

	binary.Write(b_buf, binary.LittleEndian, seed)
	key := b_buf.Bytes()
	k := int32(0)
	c := byte(0)
	for i := offset; i < len; i++ {
		k %= 4
		x := (buf[i] - c) ^ key[k]
		k++
		c = buf[i]
		buf[i] = x
	}
	return buf
}

// 加密
func DefaultNetEncrypt(seed uint32, buf []byte, offset uint32, len uint32) []byte {
	if len <= offset {
		return buf
	}
	b_buf := bytes.NewBuffer([]byte{})
	binary.Write(b_buf, binary.LittleEndian, seed)
	key := b_buf.Bytes()
	k := int32(0)
	c := byte(0)
	for i := offset; i < len; i++ {
		k %= 4
		x := (buf[i] ^ key[k]) + c
		k++
		c = x
		buf[i] = c
	}
	return buf
}

// 压缩
func GZipCompress(data []byte) []byte {
	var in bytes.Buffer
	w := gzip.NewWriter(&in)
	w.Write(data)
	w.Close()
	return in.Bytes()
}

// 解压
func GZipUnCompress(data []byte) ([]byte, error) {
	b := bytes.NewReader(data)
	r, _ := gzip.NewReader(b)
	undatas, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return undatas, nil

}

// 抄送
func CountBCC(buf []byte, offset uint32, len uint32) byte {
	value := byte(0x00)
	for i := offset; i < offset+len; i++ {
		value ^= buf[i]
	}
	return value
}

const (
	cryptA uint32 = 214013
	cryptB uint32 = 2531011
)

type HandlerFunc func(msgque IMsgQue, msg *Message) bool


// 消息处理程序
type IMsgHandler interface {
	OnNewMsgQue(msgque IMsgQue) bool                         //新的消息队列
	OnDelMsgQue(msgque IMsgQue)                              //消息队列关闭
	OnProcessMsg(msgque IMsgQue, msg *Message) bool          //默认的消息处理函数
	OnConnectComplete(msgque IMsgQue, ok bool) bool          //连接成功
	GetHandlerFunc(msgque IMsgQue, msg *Message) HandlerFunc //根据消息获得处理函数
}

// 消息注册
type IMsgRegister interface {
	// 注册
	Register(cmd, act uint8, fun HandlerFunc)
	// 注册消息
	RegisterMsg(v interface{}, fun HandlerFunc)
}


// Def的消息处理
type DefMsgHandler struct {
	msgMap map[int32]HandlerFunc
	typeMap map[reflect.Type]HandlerFunc
}

