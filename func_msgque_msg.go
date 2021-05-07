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

// 消息的最大值
var MaxMsgDataSize uint32 = 1024 * 1024

const (
	// 消息头的大小
	MsgHeadSize = 12
)

type MessageHead struct {
	Len     uint32 //数据长度
	Error   uint16 //错误码
	Cmd     uint8  //命令
	Act     uint8  //动作
	Index   uint16 //序号
	Flags   uint8  //标记
	Bcc     uint8  //加密校验
	forever bool   // 是否是永久性

}

func (r *MessageHead) Copy() *MessageHead {
	if r == nil {
		return nil
	}
	head := &MessageHead{
		Len:     r.Len,
		Error:   r.Error,
		Cmd:     r.Cmd,
		Act:     r.Act,
		Index:   r.Index,
		Flags:   r.Flags,
		Bcc:     r.Bcc,
		forever: r.forever,
	}
	return head
}

func (r *MessageHead) Bytes() []byte {
	buf := &bytes.Buffer{}
	err := binary.Write(buf, binary.BigEndian, r)
	if err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func (r *MessageHead) Tag() int {
	return Tag(r.Cmd, r.Act, r.Index)
}

func (r *MessageHead) CmdAct() int {
	return CmdAct(r.Cmd, r.Act)
}

func (r *MessageHead) String() string {
	return fmt.Sprintf("Len:%v Error:%v Cmd:%v Act:%v Index:%v Flags:%v", r.Len, r.Error, r.Cmd, r.Act, r.Index, r.Flags)
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

func (r *Message) Copy() *Message {
	msg := &Message{
		Head:       r.Head.Copy(),
		Data:       r.Data,
		IMsgParser: r.IMsgParser,
	}
	id := len(r.Data)
	if id > 0 {
		msg.Data = make([]byte, id)
		copy(msg.Data, r.Data)
	}
	return msg
}

func (r *Message) Len() uint32 {
	if r.Head != nil {
		return r.Head.Len
	}
	return 0
}

func (r *Message) Error() uint16 {
	if r.Head != nil {
		return r.Head.Error
	}
	return 0
}

func (r *Message) Tag() int {
	if r.Head != nil {
		return Tag(r.Head.Cmd, r.Head.Act, r.Head.Index)
	}
	return 0
}

func (r *Message) CmdAct() int {
	if r.Head != nil {
		return CmdAct(r.Head.Cmd, r.Head.Act)
	}
	return 0
}

func (r *Message) Cmd() uint8 {
	if r.Head != nil {
		return r.Head.Cmd
	}
	return 0
}

func (r *Message) Act() uint8 {
	if r.Head != nil {
		return r.Head.Act
	}
	return 0
}

func (r *Message) Index() uint16 {
	if r.Head != nil {
		return r.Head.Index
	}
	return 0
}

func (r *Message) Flags() uint8 {
	if r.Head != nil {
		return r.Head.Flags
	}
	return 0
}

func (r *Message) Bcc() uint8 {
	if r.Head != nil {
		return r.Head.Bcc
	}
	return 0
}

func (r *Message) Bytes() []byte {
	if r.Head != nil && r.Data == nil {
		return r.Head.Bytes()
	}
	return r.Data
}

func (r *Message) CopyTag(old *Message) *Message {
	if r.Head != nil && old.Head != nil {
		r.Head.Cmd = old.Head.Cmd
		r.Head.Act = old.Head.Act
		r.Head.Index = old.Head.Index
	}
	return r
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
			Act:   act,
			Index: index,
		},
		Data: data,
	}

}

func NewTagMsg(cmd, act uint8, index uint16) *Message {
	return &Message{
		Head: &MessageHead{
			Cmd:   cmd,
			Act:   act,
			Index: index,
		},
	}
}
