from multiprocessing import Process
import schedule
import time
import requests
import os
import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str,
                    help='http://your_ip:your_port/v1/internal/get_config')
parser.add_argument('--host_id', type=int, help='your host id')
parser.add_argument(
    '--exec_root', default="/home/wangxt/workspace/projects/flowengine/build/aarch64/bin", type=str, help='your exec dir')
parser.add_argument(
    '--model_dir', default="/home/wangxt/workspace/projects/flowengine/build/aarch64/bin", type=str, help='your exec dir')
parser.add_argument('--out_path', default="output.json", type=str, help='export the path of config file')


def run_smoke(conf):
    global exec_root
    e_path = os.path.join(exec_root, "test_det_module")
    subprocess.call([e_path,
                    '--model_dir', args.model_dir,
                     '--host_id', str(args.host_id),
                     # '--uri', conf["CameraIp"],
                     '--uri', "rtsp://user:passward@192.168.3.2:554/test",
                     '--camera_id', str(conf["CameraId"]),
                     '--result_url', "http://114.242.23.39:9400/v1/internal/receive_alarm",
                     # '--codec', conf["VideoCode"],
                     '--codec', "h265",
                     # '--height', conf["Height"],
                     '--height', str(480),
                     # '--width', conf["Width"],
                     '--width', str(640),
                     '--place', str(conf["Location"]),
                     ])


def run_hello():
    print("hello")


def read_json(path):
    # Opening JSON file
    f = open(path)
    # returns JSON object as a dictionary
    data = json.load(f)

    # Closing file
    f.close()
    return data["data"]


def check_update(d1, d2):
    # 一一对应的比较是否存在更新
    updated_ids = []
    for i, (x, y) in enumerate(zip(d1, d2)):
        if x != y:
            updated_ids.append(i)
    return updated_ids


def get_config(url, post_data):
    response = requests.post(url, json=post_data)
    content = json.loads(response.content)

    if content["code"] != "200":
        return None
    return content["data"]


def main(args):
    post_data = {
        'id': args.host_id,
    }
    datas = get_config(args.url, post_data)
    global configs, pids
    if len(datas) < 1:
        print("Warning: get config response is empty!")
        return
    if len(configs) < 1:
        # 第一次进入配置文件
        configs = datas
        for d in configs:
            if d:
                p = Process(target=worker_types[d["AlarmType"]], args=(d,))
                p.start()
                print(p.pid)
                pids.append(p.pid)
            else:
                pids.append(-1)
    else:
        assert(len(configs) == len(datas))
        updated_ids = check_update(configs, datas)
        if len(updated_ids) == 0:
            return
        for i in updated_ids:
            if pids[i] != -1:
                os.system("kill -TERM {}".format(pids[i]))
                time.sleep(10)
            p = Process(
                target=worker_types[datas[i]["AlarmType"]], args=(datas[i],))
            p.start()
            pids[i] = p.pid

    
    # os.system("kill -TERM {}".format(pids[0]))
    # p = Process(target=run_hello)
    # p.start()
    # pids[0] = p.pid


# python host_update.py --url http://114.242.23.39:9400/v1/internal/get_config --host_id 22 --out_path /home/wangxt/workspace/projects/flowengine/tests/data/output.json
if __name__ == "__main__":
    args = parser.parse_args()

    configs = []
    pids = []

    worker_types = {
        "smoke": run_smoke,
        "phone": run_smoke
    }

    exec_root = args.exec_root

    schedule.every(5).seconds.do(main, args)

    while True:
        schedule.run_pending()
        time.sleep(1)
