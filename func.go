package func_master

func IsRuning() bool {
	return stop == 0
}

func Tag(cmd, act uint8, index uint16) int {
	return int(cmd)<<16 + int(act)<<8 + int(index)
}