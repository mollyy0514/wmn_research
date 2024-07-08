#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pympler import asizeof

__all__ = [
    "getsizeof",
]

def getsizeof(obj):
    # 內存佔用空間
    size = asizeof.asizeof(obj)
    cnt = 0
    
    while size > 1000:
        size /= 1000
        cnt += 1
        
    unit_mapping = {0: 'bytes', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB', 5: 'PB'}
    unit = unit_mapping[cnt]
    size = f'{round(size, 2)} {unit}'
    
    return size
