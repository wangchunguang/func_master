package func_master

type ConfigJson struct {
	Prefix    string
	Endpoints []string
}

func NewConfigJson() *ConfigJson {
	config := &ConfigJson{}
	err := ReadConfigJson("config.json", config)
	if err != nil {
		LogError("json Parsing failed err :=", err)
		return nil
	}
	return config
}

func NewConfigToEtcd() {
	Go(func() {
		json := NewConfigJson()
		serd, _ = NewServiceDiscovery(json.Endpoints)
		err := serd.EtcdServer(json.Prefix)
		if err != nil {
			LogError("Service acquisition failed err :%s", err)
		}
	})
}
