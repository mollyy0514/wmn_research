#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import random
from .time_converter import *

def generate_hex_string(seed, length=16):
    # 設定亂數種子
    random.seed(seed)
    # 生成隨機數
    random_number = random.getrandbits(length * 4)  # 16進制的位數需要4位二進制數表示
    # 轉換為16進位制字串
    hex_string = hex(random_number)[2:]  # [2:]是因為hex()函數生成的字符串開頭是'0x'，需要去掉
    return hex_string.zfill(length)  # 確保字串長度為length，不足的話在前面補0

def figure_identity():
    figure_identity.timestamp_str = getattr(figure_identity, 'timestamp_str', datetime_to_str(epoch_to_datetime(time.time())))
    figure_identity.counter = getattr(figure_identity, 'counter', 0)
    now = time.time()
    date = "".join(figure_identity.timestamp_str[:10].split('-'))
    hms_count = "".join(figure_identity.timestamp_str[11:16].split(':')) + str(figure_identity.counter).zfill(3)
    hex_string = generate_hex_string(now, 5)
    figure_id = "_".join([date, hms_count, hex_string])
    figure_identity.counter += 1
    return date, hms_count, hex_string, figure_id

def figure_add_prefix_suffix(image_path, image_id=None, suffix='.png'):
    dirpath = os.path.dirname(image_path)
    image_label = os.path.basename(image_path)
    if image_label.endswith(suffix):
        image_label = image_label.replace(suffix, '')
    if image_id is None:
        date, hms_count, hex_string, image_id = figure_identity()
    image_path = "_".join([image_id[:8], image_label, image_id[9:]]) + suffix
    image_path = os.path.join(dirpath, image_path)
    return image_path

def model_identity():
    model_identity.timestamp_str = getattr(model_identity, 'timestamp_str', datetime_to_str(epoch_to_datetime(time.time())))
    model_identity.counter = getattr(model_identity, 'counter', 0)
    date = "".join(model_identity.timestamp_str[:10].split('-'))
    hour_minute_count = model_identity.timestamp_str[11:13] + model_identity.timestamp_str[14:16] + str(model_identity.counter).zfill(3)
    now = time.time()
    hex_string = generate_hex_string(now, 3)
    model_id = date + '_' + hour_minute_count + hex_string
    model_identity.counter += 1
    return model_id

def model_add_prefix(model_path, model_id=None):
    dirpath = os.path.dirname(model_path)
    model_label = os.path.basename(model_path)
    if model_id is None:
        model_id = model_identity()
    model_path = "_".join([model_id, model_label])
    model_path = os.path.join(dirpath, model_path)
    return model_path
