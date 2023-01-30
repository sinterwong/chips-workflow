import requests
import json
import base64

onnx_path = "/home/wangxt/workspace/projects/flowengine/tests/data/model/pre-engine/RepVGG-C0_ce_dog-cat_96x96_93.834.onnx"
with open(onnx_path, mode="rb") as rbf:
    data = rbf.read()

url = 'http://192.168.31.54:19777/convert'
post_data = {
    'idx': 0,
    'onnx_params': base64.b64encode(data).decode(encoding="utf-8")
}
x = requests.post(url, json=post_data)
