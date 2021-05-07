package func_master

import (
	"net"
	"strings"
)

// addr ip地址
// typ  基于确定的消息头
// IMsgHandler 消息处理程序的接口
// parser 解析器
// encrypt 是否加密
// 例如：StartServer("tcp://127.0.0.1:8080", MsgTypeRpc, &msgHandler{}, PbParser, true)
func StartServer(addr string, typ MsgType, handler IMsgHandler, parser *Parser, encrypt bool) error {
	addrs := strings.Split(addr, "://")
	if addrs[0] == "tcp" || addrs[0] == "all" {
		listen, err := net.Listen("tcp", addrs[1])
		if err == nil {
			LogInfo("listen on :%v", listen.Addr())
			// 初始化tcp配置
			tcpListen := NewTcpListen(listen, typ, handler, parser, addr)
			//	 设置加密
			tcpListen.SetEncrypt(encrypt)
			Go(func() {
				tcpListen.listen()
			})
		} else {
			LogError("listen on %s failed, errstr:%s", addr, err)
			return err
		}

	}
	return nil
}
