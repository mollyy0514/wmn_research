import os
import sys
import argparse
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from tqdm import tqdm
from pprint import pprint
from pytictoc import TicToc

from allutils import *
from ho_profile import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
parser.add_argument("-r", "--route", type=str, help="experiment route")
parser.add_argument("-s", "--slice", type=int, help="slice number for testing functionality")
parser.add_argument("-p", "--model_path", type=str, help="model_path")
parser.add_argument("-m", "--model_name", type=str, help="model_name")
parser.add_argument("-mm", "--direction_metrics", type=str, default='dl_lost', help="direction and metrics")
parser.add_argument("-am", "--anchor_mode", type=str, default='by_event', help="anchor mode")
parser.add_argument("-cc", "--corr_coef", type=str, default='mle', help="correlation coefficient")
parser.add_argument("-it", "--iteration", type=int, default=1, help="iteration number")
parser.add_argument("-dt", "--dataset_type", type=str, default="train", help="dataset type")
parser.add_argument("-tt", "--test_mode", action="store_true", help="test_mode")
args = parser.parse_args()

dirc_mets = args.direction_metrics
anchor_mode = args.anchor_mode
model_corr = args.corr_coef if args.corr_coef == 'mle' or args.corr_coef == 'adjust' else f'{args.corr_coef}_cc'
iter_num = args.iteration
# save_answer = args.save_answer
dataset_type = args.dataset_type
model_name = args.model_name
model_path = args.model_path
test_mode = args.test_mode

print(dirc_mets, anchor_mode, model_corr, iter_num)

if args.dates is not None:
    selected_dates = args.dates
else:
    selected_dates = data_loader(query_dates=True)
    # selected_dates = []

if args.route is not None:
    if args.route == 'all':
        selected_routes = ['BR', 'A', 'B', 'R', 'G', 'O2']
    elif 'sub' in args.route:
        rm_element = args.route[3:]
        selected_routes = ['BR', 'A', 'B', 'R', 'G', 'O2']
        selected_routes.remove(rm_element)
    else:
        selected_routes = [args.route]
else:
    selected_routes = ['BR']
# route = args.route if args.route is not None else 'BR'
route = args.route

filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)

if args.slice is not None:
    filepaths = filepaths[:args.slice]

print(selected_routes)
print(len(filepaths))

dirc_mets_list = ['dl_lost', 'ul_lost', 'dl_excl', 'ul_excl']
for dirc_mets in dirc_mets_list:
    eval = DrEval(filepaths, route, dataset_type, 
                  model_path, model_name, anchor_mode, 
                  dirc_mets=dirc_mets, model_corr=model_corr,
                  sp_columns=['type'], ts_column='Timestamp', w_size=0.01,
                  iter_num=iter_num, test_mode=test_mode)
    eval.run_hist_method(N=iter_num)
    eval.plot()


# python3 ./dr_ho_evaluate.py -r BR -dt train -p 20240719_13430002fb_new_data_sync_v2 -am by_event -it 1 -d 2024-07-17