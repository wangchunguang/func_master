package func_master

import (
	"encoding/json"
	"github.com/golang/protobuf/proto"
)

type PBModel struct{}

func (p *PBModel) DBData(v proto.Message) []byte {
	return DBData(v)
}

func (p *PBModel) DBStr(v proto.Message) string {
	return DBStr(v)
}

func (p *PBModel) PbData(v proto.Message) []byte {
	return PBData(v)
}

func (p *PBModel) PbStr(v proto.Message) string {
	return PBStr(v)
}

func (p *PBModel) ParseDBData(data []byte, v proto.Message) bool {
	return ParseDBData(data, v)
}

func (p *PBModel) ParseDBStr(str string, v proto.Message) bool {
	return ParseDBStr(str, v)
}

func (p *PBModel) ParsePbData(data []byte, v proto.Message) bool {
	return ParsePbData(data, v)
}

func (p *PBModel) ParsePbStr(str string, v proto.Message) bool {
	return ParsePbStr(str, v)
}

func PBData(v proto.Message) []byte {
	if data, err := proto.Marshal(v); err == nil {
		return data
	} else {
		LogError("proto:%v pbdata failed:%s", v, err.Error())
	}
	return nil
}

func PBStr(v proto.Message) string {
	if data, err := proto.Marshal(v); err == nil {
		return string(data)
	} else {
		LogError("proto:%v PDStr failed:%s", v, err.Error())
	}
	return ""
}

func ParsePbData(data []byte, v proto.Message) bool {
	if err := proto.Unmarshal(data, v); err == nil {
		return true
	} else {
		LogError("proto:%s struct:%s parse pbstr failed:%s", data, v, err.Error())
	}
	return false
}

func ParsePbStr(str string, v proto.Message) bool {
	if err := proto.Unmarshal([]byte(str), v); err == nil {
		return true
	} else {
		LogError("proto:%s struct:%s parse pbstr failed:%s", str, v, err.Error())
	}
	return false
}

func JsonData(v proto.Message) []byte {
	if data, err := json.Marshal(v); err != nil {
		LogError("json conversion failed byte err:%s", err.Error())
		return nil
	} else {
		return data
	}
}

func JsonStr(v proto.Message) string {
	if data, err := json.Marshal(v); err == nil {
		return string(data)
	} else {
		LogError("json conversion failed string err :%s", err.Error())
		return ""
	}
}

func ParseJsonData(data []byte, v proto.Message) bool {
	if err := json.Unmarshal(data, v); err != nil {
		LogError("json :%s parse failed err :%s", data, err.Error())
		return false
	}
	return true
}

func ParseJsonStr(str string, v proto.Message) bool {
	if err := json.Unmarshal([]byte(str), v); err != nil {
		LogError("json :%s parse failed err :%s", str, err.Error())
		return false
	}
	return true
}

// protobuf 的解析器
type pBParser struct {
	*Parser
}

// 解析C2S
func (r *pBParser) ParseC2S(msg *Message) (IMsgParser, error) {
	if msg == nil {
		return nil, ErrPBUnPack
	}
	// 某些消息不用解析
	if msg.Head.Flags&FlagNoParse > 0 {
		return nil, nil
	}
	if p, ok := r.msgMap[msg.Head.CmdAct()]; ok {
		if p.C2S() != nil {
			if err := PBUnPack(msg.Data, p.C2S()); err != nil {
				return nil, err
			}
			p.parser = r
			return &p, nil
		} else {
			return &p, nil
		}

	}
	return nil, ErrPBUnPack
}

func PBUnPack(data []byte, msg interface{}) error {
	if msg == nil {
		return ErrPBUnPack
	}
	if err := proto.Unmarshal(data, msg.(proto.Message)); err != nil {
		LogError("PBUnPack Parsing failed msg :%v err:%s", msg, err.Error())
		return ErrPBUnPack
	}
	return nil

}

func PBPack(msg interface{}) ([]byte, error) {
	if msg == nil {
		return nil, ErrPBPack
	}
	if data, err := proto.Marshal(msg.(proto.Message)); err != nil {
		LogError("PBPack :%v marshal failure err :%s ", msg, err.Error())
		return nil, err
	} else {
		return data, ErrPBPack
	}
}

func (r *pBParser) PackMsg(v interface{}) []byte {
	data, _ := PBPack(v)
	return data
}

func (r *pBParser) GetRemindMsg(err error, t MsgType) *Message {
	if t == MsgTypeMsg {
		return NewErrMsg(err)
	} else {
		return NewStrMsg(err.Error() + "\n")
	}
}
