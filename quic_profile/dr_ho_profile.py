import os
import sys
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
# import seaborn as sns

from scipy.stats import gaussian_kde
from tqdm import tqdm
from pprint import pprint
from pytictoc import TicToc
import argparse

from allutils import *
from ho_profile import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
parser.add_argument("-r", "--route", type=str, help="experiment route")
parser.add_argument("-s", "--slice", type=int, help="slice number for testing functionality")
parser.add_argument("-m", "--metrics", type=str, help="direction and metrics")
parser.add_argument("-sm", "--sr_model", type=str, help="sr model name")
parser.add_argument("-it", "--iteration", type=int, help="iteration number")
parser.add_argument("-dt", "--usage", type=str, help="dataset usage")
parser.add_argument("-sv", "--save", action="store_true", help="save answer or not")
parser.add_argument("-tm", "--test_mode", action="store_true", help="test mode or not")
args = parser.parse_args()

if args.dates is not None:
    selected_dates = args.dates
else:
    selected_dates = data_loader(query_dates=True)
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
route = args.route if args.route is not None else 'BR'
slice_num = args.slice
dirc_mets = args.metrics if args.metrics is not None else 'dl_lost'
sr_model_name = args.sr_model if args.sr_model is not None else ''
sr_model_dscp = args.sr_model if args.sr_model is not None else ''
epochs = args.iteration if args.iteration is not None else 3
dataset_type = args.usage if args.usage is not None else 'train'
save_answer = args.save
test_mode = args.test_mode

sr_model_name = '20240719_1428000117_new_data_sync_v2'
sr_model_id = sr_model_name[:19] if len(sr_model_name) > 19 else sr_model_name
sr_model_dscp = sr_model_name[20:] if len(sr_model_name) > 19 else None

filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)
print(len(filepaths))

# dirc_mets_list = ['dl_lost', 'ul_lost', 'dl_excl', 'ul_excl']
# for dirc_mets in dirc_mets_list:
#     dr_model = DrProfile(filepaths, route,
#                 sr_model_id, sr_model_dscp,
#                 dirc_mets=dirc_mets, anchor_mode="by_event", test_mode=test_mode)


# python3 ./dr_ho_profile.py -dt train -sm BR -r BR -it 3 -d 2024-07-04 2024-07-17