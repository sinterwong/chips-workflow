import requests
import base64

# onnx_path = "/open_explorer/workspace/projects/flowengine/tests/data/models/best.onnx"
onnx_path = "/open_explorer/workspace/projects/flowengine/tests/data/models/resnet-10_ce_smoking_calling_64x64_87.719.onnx"
with open(onnx_path, mode="rb") as rbf:
    data = rbf.read()


# hyper_params = """{
#     "hyper_parameter":
#         {"IoU训练阈值(iou_t)": 0.2, "Mosaic概率(mosaic)": 1.0, "cls BCELoss正样本权重(cls_pw)": 1.0, "focal_loss系数(fl_gamma)": 0.0, "iou损失系数(box)": 0.05, "上下翻转概率(flipud)": 0.0, "亮度(hsv_v)": 0.4, "余弦退火参数(lrf)": 0.01, "图像剪切(shear)": 0.0, "图像混叠概率(mixup)": 0.0, "图像缩放(scale)": 0.5, "图片复制粘贴概率(copy_paste)": 0.0, "学习率(lr0)": 0.01, "学习率动量(momentum)": 0.937, "左右翻转概率(fliplr)": 0.5,
#          "平移(translate)": 0.1, "旋转角度(degrees)": 0.0, "有无物体BCELoss正样本权重(obj_pw)": 1.0, "有无物体系数(obj)": 1.0, "权重衰减系数(weight_decay)": 0.0005, "标签与anchor的长宽比(anchor_t)": 4.0, "每个输出层的anchors数量(anchors)": 0, "类别损失系数(cls)": 0.5, "色调(hsv_h)": 0.015, "透明度(perspective)": 0.0, "预热初始学习率(warmup_bias_lr)": 0.1, "预热学习(warmup_epochs)": 3.0, "预热学习动量(warmup_momentum)": 0.8, "饱和度(hsv_s)": 0.7},
#     "model_parameter":
#         {"batch_size": 16, "device": "", "epochs": 150, "img_size": 640, "不自动调整anchor": false, "优化器": "SGD", "准确率(acc)": 0.6218074656188605, "多尺度训练": false, "损失函数": ["BCEWithLogitsLoss", "GIOU"], "模型类型": "onnx", "测试集数量": "238", "类别": ["hand"], "训练集数量": 28961, "误报率(FPR)": 0.14103283749649173, "超参数进化": 0, "预训练模型": "/workspace/projects/yolov5/weights/yolov5n.pt"}}"""

hyper_params = """{
    "algorithm": {
        "algorithm_name": "phoneCls_x3",
        "cn_algorithm": "电话识别"
    },
    "hyper_parameter": {
        "drop_rate": 0,
        "lr": 0.1,
        "lr_gamma": 0.5,
        "lr_step_size": 5,
        "momentum": 0.9,
        "warmup_step": null,
        "weight_decay": 0.0005,
        "蒸馏温度": 1
    },
    "model_parameter": {
        "batch_size": 16,
        "device": [
            0
        ],
        "epochs": 300,
        "img_size": [
            64,
            64
        ],
        "use_amp": false,
        "优化器": "sgd",
        "准确率(acc)": 0.880915003363983,
        "损失函数": "ce",
        "模型类型": "onnx",
        "测试集数量": 456,
        "类别": [
            "0",
            "smoking",
            "calling"
        ],
        "自蒸馏": false,
        "训练集数量": 8918,
        "预训练模型": "weights/resnet18-5c106cde.pth"
    }
}"""

# url = 'http://47.92.80.248:9888/convert_x3'
url = 'http://172.17.0.11:9888/convert_x3'
post_data = {
    'idx': 0,
    'onnx_params': base64.b64encode(data).decode(encoding="utf-8"),
    "hyper_params": hyper_params
}
x = requests.post(url, json=post_data)
