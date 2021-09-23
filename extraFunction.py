import json


def ipCaoncatenate(ip):
	camAddressDic = {"source": ip}
	with open('yolov5_config.json', 'r', encoding='utf8') as fp:
		opt = json.load(fp)
	for key, value in opt.items():
		camAddressDic[key] = value
	opt = camAddressDic
	return opt
