package func_master

// StartConnect 开始连接,连接的是自己内部的服务
func StartConnect(netype string, addr string, typ MsgType, handler IMsgHandler, parser *Parser, user interface{}) IMsgQue {
	if IsStop() {
		return nil
	}
	if netype == "tcp" {
		msgque := newTcpConn(netype, addr, nil, typ, handler, parser, user)
		if handler.OnNewMsgQue(msgque) {
			msgque.init = true
			if msgque.Connect() {
				return msgque
			}
			LogError("connect to:%s:%s failed", netype, addr)
		} else {
			msgque.Stop()
		}
	}
	return nil
}
