package func_master

import (
	"encoding/binary"
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
	MsgTypeRpc                //消息基于确定的消息头
)

type NetType int

const (
	NetTypeTcp NetType = iota //TCP类型
)

type ConnType int

const (
	ConnTypeListen  ConnType = iota //监听
	ConnTypeConn                    //连接产生的
	ConnTypeGateWay                 // 网关转发
	ConnTypeAccept                  //Accept产生的
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
	interactive   bool   // 是否进行三次握手成功

}

// 设置是否快速发送
func (mq *msgQue) SetSendFast() {
	mq.sendFast = true
}

// 返回通道的id
func (mq *msgQue) Id() uint32 {
	return mq.id
}

// 消息头，
func (mq *msgQue) GetMsgType() MsgType {
	return mq.msgTyp
}

// 获取是否加密
func (mq *msgQue) GetEncrypt() bool {
	return mq.encrypt
}

// 返回通道类型
func (mq *msgQue) GetConnType() ConnType {
	return mq.connTyp
}

// 是否停止
func (mq *msgQue) Available() bool {
	return mq.available
}

// 设置服务器与客户端是否进行三次握手
func (mq *msgQue) SetInteractive() {
	mq.interactive = true
}

// 发送
func (mq *msgQue) Send(m *Message) (re bool) {
	if m == nil || mq.Available() {
		return
	}
	// 如果写入里面的长度 大于设置的长度，直接返回
	if len(mq.cwrite) > cwrite_chan_len-1 {
		return
	}
	defer func() {
		if err := recover(); err != nil {
			re = false
		}
	}()
	mq.cwrite <- m
	return true
}

// 向里面添加数据
func (mq *msgQue) SendString(str string) (re bool) {
	return mq.Send(&Message{Data: []byte(str)})
}

func (mq *msgQue) SendStringLn(str string) (re bool) {
	return mq.SendString(str + "\n")

}

func (mq *msgQue) SendByteStr(str []byte) (re bool) {
	return mq.SendString(string(str))
}

func (mq *msgQue) SendByteStrLn(str []byte) (re bool) {
	return mq.SendString(string(str) + "\n")
}

func (mq *msgQue) SendCallback(m *Message, c chan *Message) (re bool) {
	if c == nil || cap(c) < 1 {
		LogError("try send callback but chan is null or no buffer")
		return
	}
	if mq.Send(m) {
		mq.setCallback(m.Tag(), c)
	} else {
		c <- nil
		return
	}
	return true
}

// 设置回调
func (mq *msgQue) setCallback(tag int, c chan *Message) {
	mq.callbackLock.Lock()
	defer func() {
		if err := recover(); err != nil {
			LogError("msgQue setCallback failure")
		}
		mq.callback[tag] = c
		mq.callbackLock.Unlock()
	}()
	if mq.callback == nil {
		mq.callback = make(map[int]chan *Message)
	}
	oc, ok := mq.callback[tag]
	if ok {
		oc <- nil
	}

}

// 设置种子
func (mq *msgQue) SetSeed(data []byte) {
	mq.oseed = uint32(Timestamp)
	mq.iseed = binary.BigEndian.Uint32(data)
}

// 设置超时
func (mq *msgQue) SetTimeout(t int) {
	if t > 0 {
		mq.timeout = t
	}
}

// 查询超时
func (mq *msgQue) GetTimeout() int {
	return mq.timeout
}

// 是否超时
func (mq *msgQue) isTimeout(tick *time.Timer) bool {
	left := int(Timestamp - mq.lastTick)
	if left < mq.timeout || mq.timeout == 0 {
		if mq.timeout == 0 {
			tick.Reset(time.Second * time.Duration(DefMsgQueTimeout))
		} else {
			tick.Reset(time.Second * time.Duration(mq.timeout-left))
		}
		return false
	}
	LogInfo("msgque close because timeout id:%v wait:%v timeout:%v", mq.id, left, mq.timeout)
	return true
}

func (mq *msgQue) Reconnect(t int) {

}

// 处理消息程序
func (mq *msgQue) GetHandler() IMsgHandler {
	return mq.handler
}

// 新增用户
func (mq *msgQue) SetUser(user interface{}) {
	mq.user = user
}

// 获取用户
func (mq *msgQue) GetUser() interface{} {
	return mq.user
}

// 尝试回调，同步逻辑处理
func (mq *msgQue) tryCallback(msg *Message) (re bool) {
	if mq.callback == nil {
		return false
	}
	defer func() {
		if err := recover(); err != nil {
			LogError(err)
		}
		mq.callbackLock.Unlock()
	}()
	mq.callbackLock.Lock()
	if mq.callback != nil {
		tag := msg.Tag()
		if c, ok := mq.callback[tag]; ok {
			delete(mq.callback, tag)
			c <- msg
			re = true
		}
	}
	return
}

func (mq *msgQue) baseStop() {
	if mq.cwrite != nil {
		close(mq.cwrite)
	}
	for k, v := range mq.callback {
		v <- nil
		delete(mq.callback, k)
	}
	msgqueMapSync.Lock()
	delete(msgqueMap, mq.id)
	msgqueMapSync.Unlock()
}

// 处理消息
func (mq *msgQue) processMsg(msgque IMsgQue, msg *Message) bool {
	//作为客户端时--robot
	if mq.connTyp == ConnTypeConn && msg.Head.Cmd == 0 && msg.Head.Act == 0 {
		if len(msg.Data) != 8 {
			LogWarn("init seed msg err")
			return false
		}
		mq.encrypt = true
		mq.oseed = binary.BigEndian.Uint32(msg.Data[:4])
		mq.iseed = binary.BigEndian.Uint32(msg.Data[4:])
		return true
	}
	// 数据经过加密
	if msg.Head != nil && msg.Head.Flags&FlagEncrypt > 0 {
		msg.Data = DefaultNetDecrypt(mq.iseed, msg.Data, 0, msg.Head.Len)
		bcc := CountBCC(msg.Data, 0, msg.Head.Len)
		//LogInfo("End Decrypt seed:%d bcc:%v Head:%v data:%v", r.iseed, bcc, msg.Head, msg.Data)
		if msg.Head.Bcc != bcc {
			LogWarn("client bcc err conn:%d", mq.id)
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

	// 当是网关的时候 网关处理逻辑
	if mq.connTyp == ConnTypeGateWay {

	}

	if mq.parser != nil {
		mp, err := mq.parser.ParseC2S(msg)
		if err == nil {
			msg.IMsgParser = mp
		} else {
			if mq.parser.GetErrType() == ParseErrTypeSendRemind {
				//机器人作为客户端时，不能全部注册消息，但不能断开连接
				if mq.connTyp == ConnTypeConn {
					return true
				}

				return false
			} else if mq.parser.GetErrType() == ParseErrTypeClose {
				return false
			} else if mq.parser.GetErrType() == ParseErrTypeContinue {
				return true
			}
		}
	}

	f := mq.handler.GetHandlerFunc(msgque, msg)
	if f == nil {
		f = mq.handler.OnProcessMsg
	}
	return f(msgque, msg)

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
	// 注册msgMap消息
	Register(cmd, act uint8, fun HandlerFunc)
	// 注册typeMap消息
	RegisterMsg(v interface{}, fun HandlerFunc)
}

// Def的消息处理
type DefMsgHandler struct {
	msgMap  map[int]HandlerFunc
	typeMap map[reflect.Type]HandlerFunc
}

func (r *DefMsgHandler) OnNewMsgQue(msgque IMsgQue) bool {
	return true
}

func (r *DefMsgHandler) OnDelMsgQue(msgque IMsgQue) {

}

func (r *DefMsgHandler) OnProcessMsg(msgque IMsgQue, msg *Message) bool {
	return true
}

func (r *DefMsgHandler) OnConnectComplete(msgque IMsgQue, ok bool) bool {
	return true
}

func (r *DefMsgHandler) GetHandlerFunc(msgque IMsgQue, msg *Message) HandlerFunc {
	// 回调不为空，直接返回正在处理的消息
	if msgque.tryCallback(msg) {
		return r.OnProcessMsg
	}
	// 获取计算动作与指令组合的数据为0
	if msg.CmdAct() == 0 {
		if r.typeMap != nil {
			if f, ok := r.typeMap[reflect.TypeOf(msg.C2S())]; ok {
				return f
			}
		}
	} else if r.msgMap != nil { // 消息map里面有初始化的消息
		if f, ok := r.msgMap[msg.CmdAct()]; ok {
			return f
		}

	}
	return nil

}

func (r *DefMsgHandler) Register(cmd, act uint8, fun HandlerFunc) {
	if r.msgMap == nil {
		r.msgMap = map[int]HandlerFunc{}
	}
	r.msgMap[CmdAct(cmd, act)] = fun
}

func (r *DefMsgHandler) RegisterMsg(v interface{}, fun HandlerFunc) {
	msgType := reflect.TypeOf(v)
	if msgType != nil && msgType.Kind() != reflect.Ptr {
		LogFatal("message pointer required")
		return
	}
	if r.typeMap == nil {
		r.typeMap = map[reflect.Type]HandlerFunc{}
	}
	r.typeMap[msgType] = fun
}

type EchoMsgHandler struct {
	DefMsgHandler
}

func (r *EchoMsgHandler) OnProcessMsg(msgque IMsgQue, msg *Message) bool {
	msgque.Send(msg)
	return true
}

type msgHandler struct {
	DefMsgHandler
	NewMsgQue       func(msgque IMsgQue) bool               //新的消息队列
	DelMsgQue       func(msgque IMsgQue)                    //消息队列关闭
	ProcessMsg      func(msgque IMsgQue, msg *Message) bool //默认的消息处理函数
	ConnectComplete func(msgque IMsgQue, ok bool) bool      //连接成功
}

func (r *msgHandler) OnNewMsgQue(msgque IMsgQue) bool {
	if r.NewMsgQue != nil {
		return r.NewMsgQue(msgque)
	}
	return true
}

func (r *msgHandler) OnDelMsgQue(msgque IMsgQue) {
	if r.DelMsgQue != nil {
		r.DelMsgQue(msgque)
	}

}

func (r *msgHandler) OnProcessMsg(msgque IMsgQue, msg *Message) bool {
	if r.ProcessMsg != nil {
		return r.ProcessMsg(msgque, msg)
	}

	return true
}
func (r *msgHandler) OnConnectComplete(msgque IMsgQue, ok bool) bool {
	if r.ConnectComplete != nil {
		return r.ConnectComplete(msgque, ok)
	}
	return true
}
