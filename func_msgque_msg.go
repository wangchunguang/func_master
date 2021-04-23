package func_master

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

type Message struct {
	Head       *MessageHead //消息头，可能为nil
	Data       []byte       //消息数据
	IMsgParser              //解析器
	User       interface{}  //用户自定义数据
}

func (r *Message) Tag() int {
	if r.Head != nil {
		return Tag(r.Head.Cmd, r.Head.Act, r.Head.Index)
	}
	return 0
}

type MessageHead struct {
	Len     uint32 //数据长度
	Error   uint16 //错误码
	Cmd     uint8  //命令
	Act     uint8  //动作
	Index   uint16 //序号
	Flags   uint8  //标记
	Bcc     uint8  //加密校验
	forever bool
	data    []byte
}
