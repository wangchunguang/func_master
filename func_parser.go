package func_master
type IMsgParser interface {
	C2S() interface{}
	S2C() interface{}
	C2SData() []byte
	S2CData() []byte
	C2SString() string
	S2CString() string
}



type ParserType int

const (
	ParserTypePB  ParserType = iota //protobuf类型，用于和客户端交互
)

type ParseErrType int

const (
	ParseErrTypeSendRemind ParseErrType = iota //消息解析失败发送提醒消息
	ParseErrTypeContinue                       //消息解析失败则跳过本条消息
	ParseErrTypeAlways                         //消息解析失败依然处理
	ParseErrTypeClose                          //消息解析失败则关闭连接
)
type ParseFunc func() interface{}

type MsgParser struct {
	s2c     interface{}
	c2s     interface{}
	c2sFunc ParseFunc
	s2cFunc ParseFunc
	parser  IParser
}

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


