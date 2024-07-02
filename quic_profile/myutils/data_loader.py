import os
import json
import pandas as pd
import itertools as it
import yaml

with open(os.path.join(os.path.dirname(__file__), "db_path.txt"), "r") as f:
    PATH_TO_DATABASE = f.readline()

def data_loader (
    mode='sr', query_dates=False, show_info=False,
    selected_dates=[], selected_exps=[], selected_routes=[],
    excluded_dates=[], excluded_exps=[], excluded_routes=[],
    root_dir=PATH_TO_DATABASE):
    
    if query_dates:
        dates = [s for s in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, s)) and s not in ['backup']]
        '''
        ...

        '''
        return dates
    
    # Collect experiments
    date_dirs = [os.path.join(root_dir, s) for s in selected_dates if s not in excluded_dates]
    exps_dict = {}
    for date_dir in date_dirs:
        # get the date folder name
        date = os.path.basename(date_dir)
        try:
            yaml_filepath = os.path.join(date_dir, f'{date}.yml')
            with open(yaml_filepath, 'r', encoding='utf-8') as yaml_file:
                my_dict = yaml.safe_load(yaml_file)
        except:
            json_filepath = os.path.join(date_dir, f'{date}.json')
            with open(json_filepath, 'r', encoding='utf-8') as json_file:
                my_dict = json.load(json_file)

    for i, (exp, item) in enumerate(my_dict.items()):
        if len(selected_exps) != 0 and exp not in selected_exps:
            continue
        if len(excluded_exps) != 0 and exp in excluded_exps:
            continue
        if len(selected_routes) != 0 and item['route'] not in selected_routes:
            continue
        if len(excluded_routes) != 0 and item['route'] in excluded_routes:
            continue
        try:
            exps_dict[date] = {**exps_dict[date], **{exp: item}}
        except:
            exps_dict[date] = {exp: item}
    
    if show_info:
        for date, exps in exps_dict.items():
            print(date, len(exps))
            for exp_name, exp in exps.items():
                print({exp_name: exp})
    
    filepaths = []
    if mode == 'sr':
        for date, exps in exps_dict.items():
            for exp_name, exp in exps.items():
                exp_dir = os.path.join(root_dir, date, exp_name)
                devices = list(exp['devices'].keys())
                try:
                    trips = ['#{:02d}'.format(s[0]) for s in exp['ods'][1:]]
                except:
                    trips = ['#{:02d}'.format(s) for s in list(exp['ods'].keys())[1:]]
                for trip in trips:
                    for dev in devices:
                        data_dir = os.path.join(exp_dir, dev, trip, 'data')
                        filepaths.append([
                                os.path.join(data_dir, 'handover_info_log.csv'),
                                os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.startswith('dl_processed_sent')][0]),
                                os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.startswith('ul_processed_sent')][0]),
                                os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.endswith('rrc.csv')][0]),
                                os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.endswith('ml1.csv') and not s.endswith('nr_ml1.csv')][0]),
                                os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.endswith('nr_ml1.csv')][0]),
                                ])
    ## TODO: elif mode == 'dr':
    return filepaths

# 
def data_aligner(df, ho_df):
    empty_data = False
    
    if df.empty or ho_df.empty:
        empty_data = True
        return df, ho_df, empty_data
    
    # ensure the handover event are all during the experiment interval and delete handover that is not during the experiments.
    start_ts = df.iloc[0]['Timestamp'] - pd.Timedelta(seconds=1)
    end_ts = df.iloc[-1]['Timestamp'] + pd.Timedelta(seconds=1)
    ho_df = ho_df[(ho_df['start'] >= start_ts) & (ho_df['start'] < end_ts)].reset_index(drop=True)
    
    if ho_df.empty:
        empty_data = True
        return df, ho_df, empty_data
    # only get the packets that start from 100 sec before the first handover event, and 100 sec after the last handover event.
    start_ts = ho_df.iloc[0]['start'] - pd.Timedelta(seconds=100)
    end_ts = ho_df.iloc[-1]['start'] + pd.Timedelta(seconds=100)
    df = df[(df['Timestamp'] >= start_ts) & (df['Timestamp'] < end_ts)].reset_index(drop=True)
    
    return df, ho_df, empty_data

