package func_master

import (
	"errors"
	message "func_master/proto"
	"reflect"
	"strconv"
	"strings"
)

type TypeKind = reflect.Kind

type funcMessage = func(v interface{})

const (
	TypeInvalid TypeKind = iota
	TypeBool
	TypeInt
	TypeInt8
	TypeInt16
	TypeInt32
	TypeInt64
	TypeUint
	TypeUint8
	TypeUint16
	TypeUint32
	TypeUint64
	TypeUintptr
	TypeFloat32
	TypeFloat64
	TypeComplex64
	TypeComplex128
	TypeArray
	TypeChan
	TypeFunc
	Interface
	TypeMap
	TypePtr
	TypeSlice
	TypeString
	TypeStruct
	TypeUnsafePointer
)

func ParseBaseKind(kind reflect.Kind, data string) (interface{}, error) {
	switch kind {
	case reflect.String:
		return data, nil
	case reflect.Bool:
		fold := strings.ToLower(data)
		v := data == "1" || data == fold
		return v, nil
	case reflect.Int:
		parseInt, err := strconv.ParseInt(data, 0, 64)
		return int(parseInt), err
	case reflect.Int8:
		x, err := strconv.ParseInt(data, 0, 8)
		return int8(x), err
	case reflect.Int16:
		x, err := strconv.ParseInt(data, 0, 16)
		return int16(x), err
	case reflect.Int32:
		x, err := strconv.ParseInt(data, 0, 32)
		return int32(x), err
	case reflect.Int64:
		x, err := strconv.ParseInt(data, 0, 64)
		return int64(x), err
	case reflect.Float32:
		x, err := strconv.ParseFloat(data, 32)
		return float32(x), err
	case reflect.Float64:
		x, err := strconv.ParseFloat(data, 64)
		return float64(x), err
	case reflect.Uint:
		x, err := strconv.ParseUint(data, 10, 64)
		return uint(x), err
	case reflect.Uint8:
		x, err := strconv.ParseUint(data, 10, 8)
		return uint8(x), err
	case reflect.Uint16:
		x, err := strconv.ParseUint(data, 10, 16)
		return uint16(x), err
	case reflect.Uint32:
		x, err := strconv.ParseUint(data, 10, 32)
		return uint32(x), err
	case reflect.Uint64:
		x, err := strconv.ParseUint(data, 10, 64)
		return uint64(x), err
	default:
		LogError("parse failed type not found type:%v data:%v", kind, data)
		return nil, errors.New("type not found")
	}
}

// ReflectStruct 将数据添加进对应的map组合
func ReflectStruct(v interface{}, fun funcMessage) {
	typeOf := reflect.TypeOf(v)
	if typeOf.Kind() != TypeStruct {
		return
	}
	nameId := message.Message_Id_value
	for i := 0; i < typeOf.NumMethod(); i++ {
		if value, ok := nameId[typeOf.Method(i).Name]; ok {
			newsMap[value] = fun
		}
	}
}
