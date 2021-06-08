package func_master

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// BytesToInt isSymbol表示是否有无符号
func BytesToInt(b []byte, isSymbol bool) (int, error) {
	if isSymbol {
		return byteToIntS(b)
	}

	return byteToIntU(b)
}

// 字节转int无符号
func byteToIntU(b []byte) (int, error) {
	if len(b) == 3 {
		b = append([]byte{0}, b...)
	}
	byteBuffer := bytes.NewBuffer(b)
	switch len(b) {
	case 1:
		var tmp uint8
		err := binary.Read(byteBuffer, binary.BigEndian, &tmp)
		return int(tmp), err
	case 2:
		var tmp uint16
		err := binary.Read(byteBuffer, binary.BigEndian, &tmp)
		return int(tmp), err
	case 3:
		var tmp uint32
		err := binary.Read(byteBuffer, binary.BigEndian, &tmp)
		return int(tmp), err
	default:
		return 0, errors.New("BytesToInt bytes lenth is invaild! err ")
	}
}

// 字节转int有符号
func byteToIntS(b []byte) (int, error) {
	if len(b) == 3 {
		b = append([]byte{0}, b...)
	}
	buffer := bytes.NewBuffer(b)
	switch len(b) {
	case 1:
		var tmp int8
		err := binary.Read(buffer, binary.BigEndian, &tmp)
		return int(tmp), err
	case 2:
		var tmp int16
		err := binary.Read(buffer, binary.BigEndian, &tmp)
		return int(tmp), err
	case 4:
		var tmp int32
		err := binary.Read(buffer, binary.BigEndian, &tmp)
		return int(tmp), err
	default:
		return 0, errors.New("BytesToInt bytes lenth is invaild")

	}
}

// IntToByte int转字节
func IntToByte(n int, b byte) ([]byte, error) {
	switch b {
	case 1:
		tmp := int8(n)
		buffer := bytes.NewBuffer([]byte{})
		fmt.Println()
		err := binary.Write(buffer, binary.BigEndian, &tmp)
		return buffer.Bytes(), err
	case 2:
		tmp := int16(n)
		buffer := bytes.NewBuffer([]byte{})
		err := binary.Write(buffer, binary.BigEndian, &tmp)
		return buffer.Bytes(), err
	case 3, 4:
		tmp := int32(n)
		buffer := bytes.NewBuffer([]byte{})
		err := binary.Write(buffer, binary.BigEndian, &tmp)
		return buffer.Bytes(), err
	}
	return nil, errors.New("IntToBytes b param is invaild")
}
