from multiprocessing import Process
import schedule
import time
import requests
import os
import json
import argparse
import subprocess
import collections
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str,
                    help='http://your_ip:your_port/v1/internal/get_config')
parser.add_argument('--host_id', type=int, help='your host id')
parser.add_argument(
    '--exec_root', default="/home/wangxt/workspace/projects/flowengine/build/aarch64/bin", type=str, help='your exec dir')
parser.add_argument(
    '--model_root', default="/home/wangxt/workspace/projects/flowengine/tests/data", type=str, help='your model dir')
parser.add_argument('--out_path', default="output.json",
                    type=str, help='export the path of config file')
parser.add_argument('--config_root', default="/home/wangxt/workspace/projects/flowengine/conf/app",
                    type=str, help='export the path of config file')

parser.add_argument('--result_url', default="http://localhost:9403/v1/internal/receive_alarm",
                    type=str, help='app send output url')


def run_app(config_path):
    e_path = os.path.join(args.exec_root, "test_pipeline")
    subprocess.call([e_path,
                     '--result_url', args.result_url,
                     '--config_path', config_path,
                     'num_workers', str(5)
                     ])


def read_json(path):
    # Opening JSON file
    f = open(path)
    # returns JSON object as a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    if data["code"] == 200:
        return data["data"]
    else:
        return None


def write_json(path, content):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def check_update(old_data, new_data):
    updated_configs = []
    o_datas = list(map(json.dumps, old_data))
    n_datas = list(map(json.dumps, new_data))
    all_datas = o_datas + n_datas
    counter = collections.Counter(all_datas)
    for k, v in counter.items():
        result = json.loads(k)
        if v == 1:
            result["Status"] = 0 if k in o_datas else 1
            updated_configs.append(result) 
    return updated_configs


def gen_json(path, datas, status=1):
    datas = copy.deepcopy(datas)
    out_template = {
        "code": 200,
        "msg": "SUCCESS"
    }
    for i, d in enumerate(datas):
        if "Status" not in datas[i]:
            datas[i]["Status"] = status
        if isinstance(d["Config"], str):
            configs = dict(json.loads(d["Config"]))
        else: 
            configs = d["Config"]
        configs["modelDir"] = args.model_root
        datas[i]["Config"] = configs
    out_template["data"] = datas
    write_json(path, out_template)


def get_stream_name(cip):
    try:
        return cip.split("@")[1].split("/")[0].replace(".", "_").replace(":", "_")
    except IndexError:
        return os.path.basename(cip)

def convert_config(datas):
    stream_mapping = {}
    for d in datas:
        key = get_stream_name(d["CameraIp"])
        stream_mapping.setdefault(key, [])
        stream_mapping[key].append(d)
    return stream_mapping


def get_config(url, post_data):
    response = requests.post(url, json=post_data)
    content = json.loads(response.content)

    if content["code"] != "200":
        return None
    return convert_config(content["data"])


def run_process(stream_name, config_path):
    global stream2pid
    p = Process(target=run_app, args=(config_path,))
    p.start()
    stream2pid[stream_name] = p.pid


def stop_process(stream_name):
    os.system("kill -s TERM {}".format(stream2pid[stream_name]))
    del stream2pid[stream_name]


def run():
    global stream2pid, stream2configs
    post_data = {
        'id': args.host_id,
    }
    # post 获取最新配置
    new_configs = get_config(args.url, post_data)

    if not os.path.exists(args.config_root):
        os.makedirs(args.config_root)
    # 1. 检查是否有原先不存在的流，如果有，开进程
    # 2. 检查是否有原先有现在没有的流，如果有，kill进程
    # 3. 检查原先存在的流是否存在更新，如果存在，更新配置文件
    for nk in new_configs.keys():
        if nk in stream2configs:
            # 原本有，现在也有
            updated_configs = check_update(stream2configs[nk], new_configs[nk])
            if len(updated_configs) > 0:
                gen_json(os.path.join(args.config_root,
                         nk + ".json"), updated_configs)
        else:
            # 原本没有，现在有
            gen_json(os.path.join(args.config_root, nk + ".json"), new_configs[nk], status=1)
            run_process(nk, os.path.join(args.config_root, nk + ".json"))
    for ok in stream2configs.keys():
        if ok not in new_configs.keys():
            # 原本有，现在没有
            stop_process(ok)
    stream2configs = new_configs

# python app_control.py --url http://localhost:9403/v1/internal/get_config --host_id 62
if __name__ == "__main__":
    args = parser.parse_args()
    stream2pid = {}
    stream2configs = {}

    schedule.every(10).seconds.do(run)

    while True:
        schedule.run_pending()
        time.sleep(1)
