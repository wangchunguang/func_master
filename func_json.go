package func_master

import (
	"encoding/json"
	"errors"
	"io/ioutil"
)

func ReadConfigJson(path string, v interface{}) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return errors.New("File parsing failed")
	}
	err = json.Unmarshal(data, v)
	if err != nil {
		return errors.New("Conversion failed")
	}
	return nil
}

func GetMarshalToJson(msg interface{}) ([]byte, error) {
	if msg == nil {
		return nil, ErrJsonPack
	}
	marshal, err := json.Marshal(msg)
	if err != nil {
		return nil, ErrJsonPack
	}
	return marshal, nil
}

func GetUnMarshalToJson(data []byte) (msg interface{}, err error) {
	if data == nil {
		return nil, ErrMsgToJson
	}
	err = json.Unmarshal(data, &msg)
	if err != nil {
		return nil, ErrJsonUnPack
	}
	return msg, err
}

func JsonUnPack(data []byte, msg interface{}) error {
	if data == nil || msg == nil {
		return ErrJsonUnPack
	}
	if err := json.Unmarshal(data, msg); err != nil {
		LogInfo("unmarshal json err:%v data:%s", err, data)
		return ErrJsonUnPack
	}
	return nil
}

func JsonPack(msg interface{}) ([]byte, error) {
	if msg == nil {
		return nil, ErrJsonPack
	}
	if data, err := json.Marshal(msg); err != nil {
		LogInfo("marshal json err:%v data:%s", err, data)
		return nil, ErrJsonUnPack
	} else {
		return data, nil
	}
}
