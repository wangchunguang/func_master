package func_master

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io/ioutil"
	"net"
)

// 网关 它负责与客户端建立连接，接收客户端发送过来的消息，并对消息进行验证，分发等
// 设计思路：
//	1. 服务器生成一个加密种子input传入客户端，为客户端的output种子
//	2. 客户端接收到加密种子之后，生成一个新的input种子，发给服务器,为服务器的output种子
// 	3. 服务器接收到客户端生成的种子，推送到客户端形成三次握手

// DefaultNetDecrypt 解密
// seed 加密解密种子，  buf 数据，开始的下标位置 长度
func DefaultNetDecrypt(seed uint32, buf []byte, offset uint32, len uint32) []byte {
	if len < offset {
		LogError("Decryption length is not enough")
		return buf
	}
	b_buf := bytes.NewBuffer([]byte{})
	binary.Write(b_buf, binary.LittleEndian, seed)
	key := b_buf.Bytes()
	k := int32(0)
	c := byte(0)
	for i := offset; i < len; i++ {
		k %= 4
		x := (buf[i] - c) ^ key[k]
		k++
		c = buf[i]
		buf[i] = x
	}
	return buf
}

// DefaultNetEncrypt 加密
func DefaultNetEncrypt(seed uint32, buf []byte, offset uint32, len uint32) []byte {
	if len <= offset {
		return buf
	}
	b_buf := bytes.NewBuffer([]byte{})
	binary.Write(b_buf, binary.LittleEndian, seed)
	key := b_buf.Bytes()
	k := int32(0)
	c := byte(0)
	for i := offset; i < len; i++ {
		k %= 4
		x := (buf[i] ^ key[k]) + c
		k++
		c = x
		buf[i] = c
	}
	return buf
}

// GZipCompress 压缩
func GZipCompress(data []byte) []byte {
	var in bytes.Buffer
	w := gzip.NewWriter(&in)
	w.Write(data)
	w.Close()
	return in.Bytes()
}

// GZipUnCompress 解压
func GZipUnCompress(data []byte) ([]byte, error) {
	b := bytes.NewReader(data)
	r, _ := gzip.NewReader(b)
	undatas, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return undatas, nil
}

// GateWayAddr 获取指定服务下，负载均衡获得的一个ip地址
func GateWayAddr(host string) net.Conn {
	if len(gateWayMap) == 0 {
		return nil
	}
	if value, ok := gateWayMap[host]; ok {
		load = &LoadBalanceServerRoundRobin{value}
		server := load.Select()
		return server.Coon
	}
	return nil

}
