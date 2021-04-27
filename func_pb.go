package func_master

import (
	"encoding/json"
	"github.com/golang/protobuf/proto"
	"github.com/vmihailenco/msgpack"
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

func DBData(v proto.Message) []byte {
	if data, err := msgpack.Marshal(v); err == nil {
		return data
	} else {
		LogError("msgpack :%v dbdata failed :%s", v, err.Error())
		return nil
	}
}

func DBStr(v proto.Message) string {
	if data, err := msgpack.Marshal(v); err == nil {
		return string(data)
	} else {
		LogError("msgpack :%s dbstring failed :%s", v, err.Error())
		return ""
	}
}

func ParseDBData(data []byte, v proto.Message) bool {
	if err := msgpack.Unmarshal(data, v); err != nil {
		LogError("msgpack parsedbdata data :%v err :%s", data, err.Error())
		return false
	}
	return true
}

func ParseDBStr(str string, v proto.Message) bool {
	if err := msgpack.Unmarshal([]byte(str), v); err != nil {
		LogError("msgpack parsedbstr str :%v err:%s", str, err.Error())
		return false
	}
	return true
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
