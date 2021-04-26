package func_master

import "github.com/golang/protobuf/proto"

func PbData(v proto.Message) []byte {
	if data, err := proto.Marshal(v); err == nil {
		return data
	} else {
		LogError("proto:%v pbdata failed:%s", v, err.Error())
	}
	return nil
}
