from flask import Flask, jsonify, request
from multiprocessing import Process
import subprocess
import os
import argparse
import base64

app = Flask(__name__)
app.config['DEBUG'] = True


parser = argparse.ArgumentParser()
parser.add_argument(
    '--exec_root', default="/home/wangxt/workspace/projects/flowengine/build/aarch64/bin", type=str, help='your exec dir')
parser.add_argument('--onnx_out_path', default="temp.onnx",
                    type=str, help='export the path of onnx model file')
parser.add_argument('--engine_out_path', default="temp.engine",
                    type=str, help='export the path of tensorrt engine file')


def __convert():
    trtexec_path = os.path.join(args.exec_root, "trtexec")
    subprocess.call([trtexec_path,
                     '--onnx=' + args.onnx_out_path,
                     '--saveEngine=' + args.engine_out_path,
                     '--fp16',
                     ])

@app.route("/convert", methods=['POST'])
def convert():
    onnx_params = request.json.get("onnx_params")  # onnx params
    idx = request.json.get("idx")  # onnx params
    with open(args.onnx_out_path,"wb") as f:
        f.write(base64.b64decode(onnx_params))
    print("***********************")
    p = Process(target=__convert)
    p.start()
    p.join()
    with open(args.engine_out_path,"rb") as f:
        data = f.read()
    return jsonify({"idx": idx, "engine": base64.b64encode(data).decode(encoding='utf-8')})


if __name__ == '__main__':
    args = parser.parse_args()
    app.run(host="192.168.31.54", port=19777, debug=False, threaded=True)
