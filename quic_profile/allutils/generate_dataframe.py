#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import datetime as dt
from typing import List
from typing import Union
from .time_converter import str_to_datetime_batch

def generate_dataframe(filepaths: Union[str, List[str]],
                       parse_dates: Union[List[str], None] = None,
                       sep=',', header='infer', index_col=None, dtype=None,
                       nrows=None, chunksize=None, usecols=None, low_memory=True):
    
    if isinstance(filepaths, str):
        try:
            df = pd.read_csv(filepaths, sep=sep, header=header, index_col=index_col, dtype=dtype,
                            nrows=nrows, chunksize=chunksize, usecols=usecols, low_memory=low_memory)
        except:
            df = pd.read_csv(filepaths, sep=sep, header=header, index_col=index_col, dtype=dtype,
                            nrows=nrows, chunksize=chunksize, usecols=usecols, low_memory=False)
        if chunksize is None and not df.empty:
            df = str_to_datetime_batch(df, parse_dates=parse_dates)
        return df
    
    dfs = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath, sep=sep, header=header, index_col=index_col, dtype=dtype,
                            nrows=nrows, chunksize=chunksize, usecols=usecols, low_memory=low_memory)
        except:
            df = pd.read_csv(filepath, sep=sep, header=header, index_col=index_col, dtype=dtype,
                            nrows=nrows, chunksize=chunksize, usecols=usecols, low_memory=False)
        if chunksize is None and not df.empty:
            df = str_to_datetime_batch(df, parse_dates=parse_dates)
        dfs.append(df)
    return dfs
