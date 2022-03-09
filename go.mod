module func_master

go 1.16

require (
	github.com/coreos/bbolt v1.3.5 // indirect
	github.com/coreos/etcd v3.3.25+incompatible
	github.com/coreos/pkg v0.0.0-20180928190104-399ea9e2e55f // indirect
	github.com/go-redis/redis v6.15.9+incompatible
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/golang/protobuf v1.5.2
	github.com/google/uuid v1.2.0 // indirect
	github.com/jingyanbin/basal v0.0.0-20220209104150-04b7dac159fe // indirect
	github.com/jingyanbin/datetime v0.0.0-20220117062442-c05dbbdbd02a
	github.com/jingyanbin/timezone v0.0.0-20220117062533-b63aefebbe91
	github.com/onsi/gomega v1.12.0 // indirect
	github.com/prometheus/client_golang v1.10.0 // indirect
	github.com/stretchr/testify v1.7.0 // indirect
	github.com/vmihailenco/msgpack v4.0.4+incompatible
	go.etcd.io/etcd v3.3.25+incompatible
	go.uber.org/zap v1.16.0 // indirect
	google.golang.org/appengine v1.6.7 // indirect
	gopkg.in/check.v1 v1.0.0-20201130134442-10cb98267c6c // indirect
)

replace google.golang.org/grpc => google.golang.org/grpc v1.26.0

replace github.com/coreos/bbolt v1.3.5 => go.etcd.io/bbolt v1.3.5
