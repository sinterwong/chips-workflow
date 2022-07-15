from flask import Flask, jsonify, request

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/configure", methods=['POST'])
def configure():
    device_id = request.json.get("device_id")  # 设备id
    device_type = request.json.get("device_type").strip()  # 设备型号

    params = {
        "threshold": 0.5,
        "region": [0, 0, 1920, 1080]
    }
    uri = "rtsp://admin:dahua2021@192.168.3.100:554/cam/realmonitor?channel=1&subtype=0"
    # uri = "/home/wangxt/workspace/projects/flow-demo/data/detection/sample_1080p_h264.mp4"
    return jsonify({"device_id": device_id, "device_type": device_type, "worker_type": 0, "uri": uri, "params": params})


if __name__ == '__main__':
    app.run('192.168.3.6', 9000)
