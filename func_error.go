package func_master

type Error struct {
	Id  uint16
	Str string
}

func (r *Error) Error() string {
	return r.Str
}

var idErrMap = map[uint16]error{}
var errIdMap = map[error]uint16{}

func NewError(str string, id uint16) *Error {
	err := &Error{
		Id:  id,
		Str: str,
	}
	idErrMap[id] = err
	errIdMap[err] = id
	return err

}

var (
	ErrOk             = NewError("正确", 0)
	ErrDBErr          = NewError("数据库错误", 1)
	ErrProtoPack      = NewError("协议解析错误", 2)
	ErrProtoUnPack    = NewError("协议打包错误", 3)
	ErrMsgPackPack    = NewError("msgpack打包错误", 4)
	ErrMsgPackUnPack  = NewError("msgpack解析错误", 5)
	ErrPBPack         = NewError("pb打包错误", 6)
	ErrPBUnPack       = NewError("pb解析错误", 7)
	ErrJsonPack       = NewError("json打包错误", 8)
	ErrJsonUnPack     = NewError("json解析错误", 9)
	ErrCmdUnPack      = NewError("cmd解析错误", 10)
	ErrMsgLenTooLong  = NewError("数据过长", 11)
	ErrMsgLenTooShort = NewError("数据过短", 12)
	ErrHttpRequest    = NewError("http请求错误", 13)
	ErrHttpUrlNil     = NewError("http请求的url为空", 14)
	ErrConfigPath     = NewError("配置路径错误", 50)
	ErrMsgToJson      = NewError("解析参数为空", 51)
	ErrMarshalToJson  = NewError("json解析错误", 52)
	ErrFileCreate     = NewError("文件创建错误", 61)
	ErrFileRead       = NewError("文件读取错误", 62)
	ErrFileWriter     = NewError("文件写入错误", 63)
	ErrDBDataType     = NewError("数据库数据类型错误", 101)
	ErrNetTimeout     = NewError("网络超时", 401)

	ErrErrIdNotFound = NewError("错误没有对应的错误码", 255)
)

var MinUserError = 256

func GetError(id uint16) error {
	if e, ok := idErrMap[id]; ok {
		return e
	}
	return ErrErrIdNotFound
}

func GetErrId(err error) uint16 {
	id, _ := errIdMap[err]
	return id

}

// JsonError 将错误进行json装换
type JsonError struct {
	Id  uint16 `json:"id"`
	Str string `json:"str"`
}

func GetErrorJson(err error) string {
	json, err := GetMarshalToJson(&Error{Id: GetErrId(err), Str: err.Error()})
	if err != nil {
		LogError(err)
		return ""
	}
	return string(json)

}
