package func_master

import (
	"github.com/golang/protobuf/proto"
	"github.com/vmihailenco/msgpack"
)

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
		LogError("msgpack Unmarshal data :%v err :%s", data, err.Error())
		return false
	}
	return true
}

func ParseDBStr(str string, v proto.Message) bool {
	if err := msgpack.Unmarshal([]byte(str), v); err != nil {
		LogError("msgpack Unmarshal str :%v err:%s", str, err.Error())
		return false
	}
	return true
}

func MsgPackUnPack(data []byte, msg interface{}) error {
	err := msgpack.Unmarshal(data, msg)
	return err
}

func MsgPackPack(msg interface{}) ([]byte, error) {
	data, err := msgpack.Marshal(msg)
	return data, err
}
