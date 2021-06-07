package func_master

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/golang/protobuf/proto"
	"unsafe"
)

const (
	FlagEncrypt  = 1 << 0 //数据是经过加密的
	FlagCompress = 1 << 1 //数据是经过压缩的
	FlagContinue = 1 << 2 //消息还有后续
	FlagNeedAck  = 1 << 3 //消息需要确认
	FlagAck      = 1 << 4 //确认消息
	FlagReSend   = 1 << 5 //重发消息
	FlagClient   = 1 << 6 //消息来自客服端，用于判断index来之服务器还是其他玩家
	FlagNoParse  = 1 << 7 //消息不解析
)

// MaxMsgDataSize 消息的最大值
var MaxMsgDataSize uint32 = 1024 * 1024

const (
	// MsgHeadSize 消息头的大小
	MsgHeadSize = 16
)

type MessageHead struct {
	Len     uint32 //数据长度
	Error   uint16 //错误码
	Cmd     uint8  //命令
	Index   uint16 //序号
	Flags   uint8  //标记
	Bcc     uint8  //加密校验
	Forward bool   // 是否通过网关请求转发
}

func (mh *MessageHead) FastBytes(data []byte) []byte {
	phead := (*MessageHead)(unsafe.Pointer(&data[0]))
	phead.Len = mh.Len
	phead.Error = mh.Error
	phead.Cmd = mh.Cmd
	phead.Index = mh.Index
	phead.Flags = mh.Flags
	phead.Bcc = mh.Bcc
	phead.Forward = mh.Forward
	return data

}

func (mh *MessageHead) Copy() *MessageHead {
	if mh == nil {
		return nil
	}
	head := &MessageHead{
		Len:     mh.Len,
		Error:   mh.Error,
		Cmd:     mh.Cmd,
		Index:   mh.Index,
		Flags:   mh.Flags,
		Bcc:     mh.Bcc,
		Forward: mh.Forward,
	}
	return head
}

func (mh *MessageHead) Bytes() []byte {
	buf := &bytes.Buffer{}
	err := binary.Write(buf, binary.BigEndian, mh)
	if err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func (mh *MessageHead) Tag() int {
	return Tag(mh.Cmd, mh.Index)
}

func (mh *MessageHead) String() string {
	return fmt.Sprintf("Len:%v Error:%v Cmd:%v  Index:%v Flags:%v", mh.Len, mh.Error, mh.Cmd, mh.Index, mh.Flags)
}

// 解析请求头
func NewMessageHead(data []byte) *MessageHead {
	if len(data) < MsgHeadSize {
		LogError("Insufficient data length")
		return nil
	}
	head := (*MessageHead)(unsafe.Pointer(&data))
	return head
}

type Message struct {
	Head       *MessageHead //消息头，可能为nil
	Data       []byte       //消息数据
	IMsgParser              //解析器
	User       interface{}  //用户自定义数据
}

func (m *Message) Copy() *Message {
	msg := &Message{
		Head:       m.Head.Copy(),
		Data:       m.Data,
		IMsgParser: m.IMsgParser,
	}
	id := len(m.Data)
	if id > 0 {
		msg.Data = make([]byte, id)
		copy(msg.Data, m.Data)
	}
	return msg
}

func (m *Message) Len() uint32 {
	if m.Head != nil {
		return m.Head.Len
	}
	return 0
}

func (m *Message) Error() uint16 {
	if m.Head != nil {
		return m.Head.Error
	}
	return 0
}

func (m *Message) Tag() int {
	if m.Head != nil {
		return Tag(m.Head.Cmd, m.Head.Index)
	}
	return 0
}

func (m *Message) Cmd() uint8 {
	if m.Head != nil {
		return m.Head.Cmd
	}
	return 0
}

func (m *Message) Index() uint16 {
	if m.Head != nil {
		return m.Head.Index
	}
	return 0
}

func (m *Message) Flags() uint8 {
	if m.Head != nil {
		return m.Head.Flags
	}
	return 0
}

func (m *Message) Bcc() uint8 {
	if m.Head != nil {
		return m.Head.Bcc
	}
	return 0
}

func (m *Message) Bytes() []byte {
	if m.Head != nil && m.Data == nil {
		return m.Head.Bytes()
	}
	return m.Data
}

func (m *Message) CopyTag(old *Message) *Message {
	if m.Head != nil && old.Head != nil {
		m.Head.Cmd = old.Head.Cmd
		m.Head.Index = old.Head.Index
	}
	return m
}

func NewErrMsg(err error) *Message {
	errcode, ok := errIdMap[err]
	if !ok {
		errcode = errIdMap[ErrErrIdNotFound]
	}
	return &Message{
		Head: &MessageHead{Error: errcode},
	}
}

func NewStrMsg(str string) *Message {
	return &Message{Data: []byte(str)}
}

func NewDataMsg(data []byte) *Message {
	return &Message{Head: &MessageHead{Len: uint32(len(data))},
		Data: data}
}

func NewPbMsg(msg proto.Message) *Message {
	return NewDataMsg(PBData(msg))
}

func NewMsg(cmd, act uint8, index, err uint16, data []byte) *Message {

	return &Message{
		Head: &MessageHead{
			Len:   uint32(len(data)),
			Error: err,
			Cmd:   cmd,
			Index: index,
		},
		Data: data,
	}

}

func NewTagMsg(cmd, act uint8, index uint16) *Message {
	return &Message{
		Head: &MessageHead{
			Cmd:   cmd,
			Index: index,
		},
	}
}
