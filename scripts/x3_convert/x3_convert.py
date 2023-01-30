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
    '--exec_root', default="/usr/local/bin", type=str, help='your exec dir')
parser.add_argument('--onnx_out_path', default="temp.onnx",
                    type=str, help='export the path of onnx model file')
parser.add_argument('--config_path', default="./config.yaml",
                    type=str, help='export the path of tensorrt engine file')


def __convert():
    trtexec_path = os.path.join(args.exec_root, "hb_mapper")
    subprocess.call([trtexec_path, "makertbin",
                     '--config', args.config_path,
                     '--model-type', "onnx",
                     ])


@app.route("/convert", methods=['POST'])
def convert():
    onnx_params = request.json.get("onnx_params")  # onnx params
    hyper_params = request.json.get("hyper_params")  # onnx params
    model_parameter = dict(hyper_params)["model_parameter"]
    img_size = model_parameter["img_size"]
    if isinstance(img_size, int):
        img_size = [img_size, img_size]
    idx = request.json.get("idx")  # onnx params
    with open(args.onnx_out_path, "wb") as f:
        f.write(base64.b64decode(onnx_params))
    data_process_cmd = "python data_preprocess.py --input_shape {},{} --src_dir /open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/coco --dst_dir ./calibration_data_rgb_f32_coco --pic_ext .rgb --read_mode opencv".format(
        img_size[0], img_size[1])
    os.system(data_process_cmd)

    p = Process(target=__convert)
    p.start()
    p.join()
    with open("temp_nv12.bin", "rb") as f:
        data = f.read()
    return jsonify({"idx": idx, "engine": base64.b64encode(data).decode(encoding='utf-8')})


if __name__ == '__main__':
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=19778, debug=False, threaded=True)
