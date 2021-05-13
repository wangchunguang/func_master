package func_master

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net"
	"net/http"
	"net/smtp"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	APPLOCATION_XHTML_XML  = "application/xhtml+xml"
	APPLOCATION_XML        = "application/xml"
	APPLOCATION_ATOM_XML   = "application/atom+xml"
	APPLOCATION_JSON       = "application/json"
	APPLOCATION_PDF        = "application/pdf"
	APPLOCATION_MSWORD     = "application/msword"
	APPLOCATION_STREAM     = "application/octet-stream"
	APPLOCATION_URLENCODED = "application/x-www-form-urlencoded"
	MULTIPART_FROM_DATA    = "multipart/form-data"
	POST                   = "post"
	GET                    = "get"
	TIMEOUT                = 5
	DEADLINE               = 5
)

const (

	// MsgClientSendSize 客户端加密种子的大小
	MsgClientSendSize = 4
	// MsgSendSize 加密种子的大小
	MsgSendSize = 8
)

// http get请求
func HttpGet(url string) (string, error, *http.Response) {
	if len(url) == 0 {
		return "", ErrHttpUrlNil, nil
	}
	response, err := get(url)
	defer response.Body.Close()
	if err != nil {
		return "", err, nil
	}
	all, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return "", ErrHttpRequest, response
	}
	return string(all), nil, response
}

func get(url string) (*http.Response, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, ErrHttpRequest
	}
	return resp, nil
}

// http post请求
func HttpPost(url, applocation string, data []byte) (string, error, *http.Response) {
	if len(url) == 0 {
		return "", ErrHttpUrlNil, nil
	}
	resp, err := http.Post(url, applocation, bytes.NewReader(data))
	if err != nil {
		return "", ErrHttpRequest, resp
	}
	readAll, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", ErrHttpRequest, resp
	}
	resp.Body.Close()
	return string(readAll), nil, resp
}

// http Post设置超时请求
func HttpPostTimeOut(url string, application string, data []byte) (string, error, *http.Response) {
	if len(url) == 0 {
		return "", ErrHttpUrlNil, nil
	}
	cli := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (conn net.Conn, err error) {
			dialer := net.Dialer{Timeout: time.Second * time.Duration(TIMEOUT), // 超时连接
				Deadline: time.Now().Add(time.Duration(DEADLINE)), // 发送数据超时
			}
			return dialer.DialContext(ctx, network, url)
		},
	}
	request, err := http.NewRequest(POST, url, bytes.NewReader(data))
	if err != nil {
		return "", ErrHttpRequest, nil
	}
	request.Header.Set("Content-Type", application)
	defer cli.CloseIdleConnections()
	client := &http.Client{Transport: cli}
	response, err := client.Do(request)
	if err != nil {
		return "", ErrHttpRequest, response
	}
	all, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return "", ErrHttpRequest, response
	}

	response.Body.Close()
	return string(all), nil, response
}

// HttpGetTimeOut http get设置超时请求
func HttpGetTimeOut(url string) (string, error, *http.Response) {
	if len(url) == 0 {
		return "", ErrHttpUrlNil, nil
	}
	client := http.Client{Timeout: DEADLINE}
	request, err := http.NewRequest(GET, url, nil)
	if err != nil {
		return "", ErrHttpRequest, nil
	}
	response, err := client.Do(request)
	defer response.Body.Close()
	if err != nil {
		return "", ErrHttpRequest, response
	}
	all, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return "", ErrHttpRequest, response
	}

	return string(all), nil, response
}

// HttpUpload 上传文件
func HttpUpload(url, field, file string) (*http.Response, error) {
	var lock sync.RWMutex
	lock.Lock()
	defer lock.Unlock()
	buf := new(bytes.Buffer)
	writer := multipart.NewWriter(buf)
	// 根据提供的字段名与文件名创建文件
	formFile, err := writer.CreateFormFile(field, file)
	if err != nil {
		LogError("create from file failed:%s\n", err)
		return nil, ErrFileCreate
	}
	// 打开文件
	open, err := os.Open(file)
	defer open.Close()
	if err != nil {
		LogError("pen source file failed:%s\n", err)
		return nil, ErrFileWriter
	}
	_, err = io.Copy(formFile, open)
	if err != nil {
		LogError("File write failed :%s\n", err)
		return nil, ErrFileWriter
	}
	contentType := writer.FormDataContentType()
	writer.Close()
	resp, err := http.Post(url, contentType, buf)
	if err != nil {
		LogError("Failed to upload file :%s\n", err)
		return resp, ErrHttpRequest
	}
	return resp, nil
}

// HttpDownload 下载文件
func HttpDownload(url, file string) error {
	response, err := http.Get(url)
	defer response.Body.Close()
	//	 创建文件
	create, err := os.Create(file)
	defer create.Close()
	if err != nil {
		LogError("Failed to create file %s\n", err)
		return ErrFileRead
	}
	_, err = io.Copy(create, response.Body)
	if err != nil {
		LogError("File write failed %s\n", err)
		return err
	}
	return nil
}

// SendMail 发送邮件
func SendMail(user, password, host, to, subject, body, mailtype string) error {
	hp := strings.Split(host, ":")
	auth := smtp.PlainAuth("", user, password, hp[0])
	var content_type string
	if mailtype == "html" {
		content_type = "Content-Type: text/" + mailtype + "; charset=UTF-8"
	} else {
		content_type = "Content-Type: text/plain" + "; charset=UTF-8"
	}

	msg := []byte("To: " + to + "\r\nFrom: " + user + ">\r\nSubject: " + "\r\n" + content_type + "\r\n\r\n" + body)
	send_to := strings.Split(to, ";")
	err := smtp.SendMail(host, auth, user, send_to, msg)
	return err
}
