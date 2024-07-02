#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from typing import List

def mean_downsample(data: List, sample_size=100000):
    """
    平均下採樣函數
    
    Args:
    data: 原始數據的列表
    sample_size: 下採樣後的樣本大小
    
    Returns:
    downsampled_data: 下採樣後的數據列表
    """
    chunk_size = len(data) // sample_size
    if chunk_size == 0:
        return data
    data = sorted(data)
    downsampled_data = [sum(data[i:i+chunk_size]) / chunk_size for i in range(0, len(data), chunk_size)]
    return downsampled_data

def median_downsample(data: List, sample_size=100000):
    """
    中位數下採樣函數
    
    Args:
    data: 原始數據的列表
    sample_size: 下採樣後的樣本大小
    
    Returns:
    downsampled_data: 下採樣後的數據列表
    """
    chunk_size = len(data) // sample_size
    if chunk_size == 0:
        return data
    data = sorted(data)
    downsampled_data = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        # median_index = len(chunk) // 2
        median_value = np.median(chunk)
        downsampled_data.append(median_value)
    return downsampled_data
