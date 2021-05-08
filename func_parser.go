package func_master

import "reflect"

type IMsgParser interface {
	C2S() interface{}
	S2C() interface{}
	C2SData() []byte
	S2CData() []byte
	C2SString() string
	S2CString() string
}

// MsgParser 消息解析器
type MsgParser struct {
	s2c     interface{}
	c2s     interface{}
	c2sFunc ParseFunc
	s2cFunc ParseFunc
	parser  IParser
}

func (mp *MsgParser) C2S() interface{} {
	if mp.c2s == nil && mp.c2sFunc != nil {
		mp.c2s = mp.c2sFunc
	}
	return mp.c2s
}

func (mp *MsgParser) S2C() interface{} {
	if mp.s2c == nil && mp.s2cFunc != nil {
		mp.s2c = mp.s2cFunc
	}
	return mp.s2c
}

func (mp *MsgParser) C2SData() []byte {
	return mp.parser.PackMsg(mp.C2S())
}

func (mp *MsgParser) S2CData() []byte {
	return mp.parser.PackMsg(mp.S2C())
}

func (mp *MsgParser) C2SString() string {
	return string(mp.C2SData())
}

func (mp *MsgParser) S2CString() string {
	return string(mp.S2CData())
}

type ParserType int

const (
	ParserTypePB ParserType = iota //protobuf类型，用于和客户端交互
)

type ParseErrType int

const (
	ParseErrTypeSendRemind ParseErrType = iota //消息解析失败发送提醒消息
	ParseErrTypeContinue                       //消息解析失败则跳过本条消息
	ParseErrTypeAlways                         //消息解析失败依然处理
	ParseErrTypeClose                          //消息解析失败则关闭连接
)

type ParseFunc func() interface{}

type IParser interface {
	GetType() ParserType
	GetErrType() ParseErrType
	ParseC2S(msg *Message) (IMsgParser, error)
	PackMsg(v interface{}) []byte
	GetRemindMsg(err error, t MsgType) *Message
}

type Parser struct {
	Type    ParserType
	ErrType ParseErrType
	msgMap  map[int]MsgParser
	parser  IParser
}

func (p *Parser) Get() IParser {
	if p.parser == nil {
		p.parser = &pBParser{Parser: p}
	}
	return p.parser
}

func (p *Parser) GetType() ParserType {
	return p.Type
}

func (p *Parser) GetErrType() ParseErrType {
	return p.ErrType
}

func (p *Parser) RegisterFunc(cmd, act uint8, c2sFunc ParseFunc, s2cFunc ParseFunc) {
	if p.msgMap == nil {
		p.msgMap = map[int]MsgParser{}
	}
	p.msgMap[CmdAct(cmd, act)] = MsgParser{c2sFunc: c2sFunc, s2cFunc: s2cFunc}
}

// 寄存器
func (p *Parser) Register(cmd, act uint8, c2s interface{}, s2c interface{}) {
	if p.msgMap == nil {
		p.msgMap = map[int]MsgParser{}
	}
	pm := MsgParser{}
	if c2s != nil {
		c2sType := reflect.TypeOf(c2s).Elem()
		pm.c2sFunc = func() interface{} {
			return reflect.New(c2sType).Interface()
		}
	}
	if s2c != nil {
		s2cType := reflect.TypeOf(s2c).Elem()
		pm.s2cFunc = func() interface{} {
			return reflect.New(s2cType).Interface()
		}
	}
	p.msgMap[CmdAct(cmd, act)] = pm
}
