import onnxruntime
import numpy as np
import cv2


def get_output_name(session):
    """
    output_name = session.get_outputs()[0].name
    :param session:
    :return:
    """
    output_name = []
    for node in session.get_outputs():
        output_name.append(node.name)
    return output_name


def get_input_name(session):
    """
    input_name = session.get_inputs()[0].name
    :param session:
    :return:
    """
    input_name = []
    for node in session.get_inputs():
        input_name.append(node.name)
    return input_name


def get_input_feed(input_name, image_numpy):
    """
    input_feed={input_name: image_numpy}
    :param input_name:
    :param image_numpy:
    :return:
    """
    input_feed = {}
    for name in input_name:
        input_feed[name] = image_numpy
    return input_feed


def preprocessing(image, input_size, aspect_ratio=True, alpha=255.0, beta=0):
    rh, rw = None, None
    h, w, _ = image.shape
    if aspect_ratio:
        assert(input_size[0] == input_size[1])
        data = np.zeros(
            [input_size[0], input_size[0], 3], dtype=np.float32)
        ration = float(input_size[0]) / max(h, w)
        ih, iw = round(h * ration), round(w * ration)
        data[:ih, :iw, :] = cv2.resize(image, (iw, ih)).astype(np.float32)
        rh, rw = ration, ration
    else:
        rw = input_size[0] / w
        rh = input_size[1] / h
        data = cv2.resize(
            image, (input_size[0], input_size[1])).astype(np.float32)

    if alpha != 0:
        data /= alpha

    if beta != 0:
        data -= beta
    data = data.transpose([2, 0, 1])
    return np.expand_dims(data, 0), rw, rh


def xywh2xyxy(x, padw=32, padh=32):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)  # top left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)  # bottom right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y


def box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


def nms(boxes, scores, iou_thres):
    keep_index = []
    order = scores.argsort()[::-1]
    areas = box_area(boxes)

    while order.size > 1:
        i = order[0]  # 取最大得分
        keep_index.append(i)

        # 计算最高得分与剩余矩形框的相较区域
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter) + 1e-8

        # 保留小于阈值的box
        inds = np.where(iou <= iou_thres)[0]

        # 注意这里索引加了1,因为iou数组的长度比order数组的长度少一个
        order = order[inds + 1]
    return np.array(keep_index, dtype=np.int32)


def non_max_suppression(prediction, conf_thres=0.2, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
        detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    xc = prediction[:, :, 4] > conf_thres
    nc = prediction.shape[2] - 5

    multi_label = nc > 1

    max_wh = 4096
    max_det = 300
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)
        else:
            conf = x[:, 5:].max(1, keepdims=True)
            j = np.zeros_like(conf)
            x = np.concatenate((box, conf, j), 1)[
                conf.reshape(-1) > conf_thres]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 区别每个类别的框, 达成同时做NMS
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
    return output


def assd_test():
    model_path = "/home/wangxt/workspace/projects/flowengine/tests/data/model/assd_shoulder.onnx"
    im_path = "/home/wangxt/workspace/projects/flowengine/tests/data/pedestrian.jpg"
    session = onnxruntime.InferenceSession(model_path)
    input_name = get_input_name(session)
    output_name = get_output_name(session)

    RF_half = [55.5, 71.5, 79.5]
    receptive_field_stride = [8, 8, 8]
    receptive_field_stride_start = [7, 7, 7]
    score_threshold = 0.4
    iou_threshold = 0.4

    img = cv2.imread(im_path)
    h, w, _ = img.shape
    data, rw, rh = preprocessing(img, (967, 543), False, 0, 0)
    input_feed = get_input_feed(input_name, data)
    forward_start = cv2.getTickCount()
    outputs = session.run(output_name, input_feed=input_feed)[1:]
    forward_end = cv2.getTickCount()
    print("推理耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))

    post_start = cv2.getTickCount()
    bboxes = []
    for i in range(len(outputs) - 1):
        predict = outputs[i][0]
        # show_img = (predict[0] * 255).astype(np.uint8)
        # cv2.imwrite("out_binary_{}.jpg".format(i), cv2.resize(show_img, (w, h)))
        score_map = predict[0]
        roi_indies = np.argwhere(score_map > score_threshold)
        for bi in roi_indies:
            center_x = bi[1] * receptive_field_stride[i] + receptive_field_stride_start[i]
            center_y = bi[0] * receptive_field_stride[i] + receptive_field_stride_start[i]
            xx1 = center_x - predict[1, bi[0], bi[1]] * RF_half[i]
            yy1 = center_y - predict[2, bi[0], bi[1]] * RF_half[i]
            xx2 = center_x - predict[3, bi[0], bi[1]] * RF_half[i]
            yy2 = center_y - predict[4, bi[0], bi[1]] * RF_half[i]
            x1 = np.min([xx1, xx2])
            x2 = np.max([xx1, xx2])
            y1 = np.min([yy1, yy2])
            y2 = np.max([yy1, yy2])
            w = x2 - x1
            h = y2 - y1
            bboxes.append([x1 + w / 2.0, y1 + h / 2.0, w, h, score_map[bi[0], bi[1]], 1.0])

    bboxes = np.array([bboxes])
    out = non_max_suppression(bboxes, score_threshold, iou_threshold)[0]
    out[:, 0] = out[:, 0] / rw
    out[:, 1] = out[:, 1] / rh
    out[:, 2] = out[:, 2] / rw
    out[:, 3] = out[:, 3] / rh

    out[out[:, 0] < 0] = 0
    out[out[:, 1] < 0] = 0
    out[out[:, 2] < 0] = 0
    out[out[:, 3] < 0] = 0
    out = out.astype(np.int32)
    post_end = cv2.getTickCount()
    print("后处理耗时：{}s".format((post_end - post_start) / cv2.getTickFrequency()))

    for i, dr in enumerate(out):
        cv2.rectangle(img, (dr[0], dr[1]), (dr[2], dr[3]), 255, 2, 1)
    cv2.imwrite("out.jpg", img)


if __name__ == "__main__":
    assd_test()
