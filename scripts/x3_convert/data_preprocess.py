# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import sys
sys.path.append('.')
sys.path.append(
    "/open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/01_common/python/data/")
from dataset import CifarDataset
from dataloader import DataLoader
import argparse
import cv2
import skimage.io
from preprocess import calibration_transformers
import numpy as np
import click
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_shape', type=str, help='input shape')

parser.add_argument('--src_dir', type=str, help='calibration source file')
parser.add_argument('--dst_dir', type=str, help='generated calibration file')
parser.add_argument('--pic_ext',
                    type=str,
                    default=".cali",
                    help='picture extension.')
parser.add_argument('--read_mode',
                    type=str,
                    default="opencv",
                    help='picture extension.')
parser.add_argument('--cal_img_num', type=int,
                    default=100, help='cali picture num.')

args = parser.parse_args()

transformers = calibration_transformers(list(map(int, args.input_shape.split(","))))

regular_process_list = [
    ".rgb",
    ".rgbp",
    ".bgr",
    ".bgrp",
    ".yuv",
    ".feature",
    ".cali",
]


def read_image(src_file, read_mode):
    if read_mode == "skimage":
        image = skimage.img_as_float(skimage.io.imread(src_file)).astype(
            np.float32)
    elif read_mode == "opencv":
        image = cv2.imread(src_file)
    else:
        raise ValueError(f"Invalid read mode {read_mode}")
    if image.ndim != 3:  # expend gray scale image to three channels
        image = image[..., np.newaxis]
        image = np.concatenate([image, image, image], axis=-1)
    return image


def regular_preprocess(src_file, transformers, dst_dir, pic_ext, read_mode):
    image = [read_image(src_file, read_mode)]
    for trans in transformers:
        image = trans(image)

    filename = os.path.basename(src_file)
    short_name, ext = os.path.splitext(filename)
    pic_name = os.path.join(dst_dir, short_name + pic_ext)
    print("write:%s" % pic_name)
    dtype = np.float32 if dst_dir.endswith("_f32") else np.uint8
    image[0].astype(dtype).tofile(pic_name)


def cifar_preprocess(src_file, data_loader, dst_dir, pic_ext, cal_img_num):
    for i in range(cal_img_num):
        image, label = next(data_loader)
        filename = os.path.basename(src_file)
        pic_name = os.path.join(dst_dir + '/' + str(i) + pic_ext)
        print("write:%s" % pic_name)
        image[0].astype(np.uint8).tofile(pic_name)


def main():
    '''A Tool used to generate preprocess pics for calibration.'''
    pic_num = 0
    os.makedirs(args.dst_dir, exist_ok=True)
    if args.pic_ext.strip().split('_')[0] in regular_process_list:
        print("regular preprocess")
        for src_name in sorted(os.listdir(args.src_dir)):
            pic_num += 1
            if pic_num > args.cal_img_num:
                break
            src_file = os.path.join(args.src_dir, src_name)
            regular_preprocess(src_file, transformers, args.dst_dir, args.pic_ext,
                               args.read_mode)
    elif args.pic_ext.strip().split('_')[0] == ".cifar":
        print("cifar preprocess")
        data_loader = DataLoader(CifarDataset(args.src_dir), transformers, 1)
        cifar_preprocess(args.src_dir, data_loader, args.dst_dir, args.pic_ext, args.cal_img_num)
    else:
        raise ValueError(f"invalid pic_ext {args.pic_ext}")


if __name__ == '__main__':
    main()
