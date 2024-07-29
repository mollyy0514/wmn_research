import os
import csv
import pickle
import random
import copy
import numpy as np
import pandas as pd
import portion as P
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from allutils import *

class DrEval:
    def __init__(
        self, filepaths, route, dataset_type,
        model_path, model_name, anchor_mode,
        dirc_mets='dl_lost', model_corr='mle',
        sp_columns=['type'], ts_column='Timestamp', w_size=0.01,
        iter_num=1, test_mode=False
    ):

        self.iter_num = iter_num
        self.test_mode = test_mode
        self.anchor_mode = anchor_mode
        self.sp_columns = sp_columns[:]
        self.ts_column = ts_column
        self.w_size = w_size
        
        self.dirc_mets = dirc_mets
        self.dirc, self.mets = dirc_mets[:2], dirc_mets[-4:]
        self.DIRC_TYPE = 'Downlink' if self.dirc == 'dl' else 'Uplink'
        self.METS_TYPE = 'Packet Loss' if self.mets == 'lost' else 'Excessive Latency'
        self.RATE_TYPE = 'PLR' if self.mets == 'lost' else 'ELR'
        
        self.filepaths = copy.deepcopy(filepaths)
        self.model_corr = model_corr
        self.dataset_type = dataset_type
        
        self.model_path = model_path
        self.model_name = model_name if model_name is not None else route
        self.route = route if route is not None else model_name.replace('sub', '')
        
        # self.sr_load_path = os.path.join(self.model_path, 'train', 'sr', self.dirc_mets, 'models', self.model_name)
        # self.dr_load_path = os.path.join(self.model_path, 'train', f'dr_anchor_{self.anchor_mode}', self.dirc_mets, 'models', self.model_name)
        self.sr_load_path = os.path.join(self.model_path, self.model_name, 'train', 'sr', self.dirc_mets, 'models')
        self.dr_load_path = os.path.join(self.model_path, self.model_name, 'train', f'dr_anchor_{self.anchor_mode}', self.dirc_mets, 'models')
        self.save_path = os.path.join(self.model_path, self.model_name, f'dr_anchor_{self.anchor_mode}', self.dirc_mets, self.dataset_type)
        
        print(self.save_path)
        print(self.sr_load_path)
        print(self.dr_load_path)
        
        try:
            with open(f'{self.sr_load_path}/{self.route}_kde_models.pkl', 'rb') as f:
                self.kde_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.sr_load_path}/{self.route}_hist_models.pkl', 'rb') as f:
                self.hist_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.sr_load_path}/{self.route}_scope_models.pkl', 'rb') as f:
                self.scope_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.sr_load_path}/{self.route}_plr_models.pkl', 'rb') as f:
                self.plr_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.sr_load_path}/{self.route}_sr_prob_models.pkl', 'rb') as f:
                self.sr_prob_models = pickle.load(f)[self.dirc_mets]
        except:
            with open(f'{self.sr_load_path}/{self.route}_kde_models.pkl', 'rb') as f:
                self.kde_models = pickle.load(f)
            with open(f'{self.sr_load_path}/{self.route}_hist_models.pkl', 'rb') as f:
                self.hist_models = pickle.load(f)
            with open(f'{self.sr_load_path}/{self.route}_scope_models.pkl', 'rb') as f:
                self.scope_models = pickle.load(f)['result']
            with open(f'{self.sr_load_path}/{self.route}_plr_models.pkl', 'rb') as f:
                self.plr_models = pickle.load(f)
            with open(f'{self.sr_load_path}/{self.route}_sr_prob_models.pkl', 'rb') as f:
                self.sr_prob_models = pickle.load(f)
        
        dr_prob_models_filepath = f'{self.dr_load_path}/{self.route}_dr_prob_models_{self.model_corr}.pkl'
        try:
            with open(dr_prob_models_filepath, 'rb') as f:
                self.dr_prob_models = pickle.load(f)[self.dirc_mets]
        except:
            with open(dr_prob_models_filepath, 'rb') as f:
                self.dr_prob_models = pickle.load(f)
        
        self.date, self.hms_count, self.hex_string, self.figure_id = figure_identity()

        if self.model_corr is None:
            self.save_name = f'{self.model_name}_{self.route}_mle'
        else:
            self.save_name = f'{self.model_name}_{self.route}_{self.model_corr}'
        
        self.records = []

    @staticmethod
    def generate_random_boolean(probability_true):
        return random.random() < probability_true
    
    @staticmethod
    def interpolate(x, y, ratio=0.5):
        """
        Args:
            x, y (datetime.datetime): x < y
            ratio (float): a decimal numeral in a range [0, 1]; 0 means break at x, 1 means break at y.
        Returns:
            (datetime.datetime): breakpoint of interpolation
        """
        return x + (y - x) * ratio
    
    def hist_method_anchor(self, df, ho_df):
        mets, RATE_TYPE = self.mets, self.RATE_TYPE
        scope = self.scope_models
        hist_model = self.hist_models
        prob_model = self.sr_prob_models
        
        this_df = df.copy()
        for i, row in ho_df.iterrows():
            prior_row = ho_df.iloc[i-1] if i != 0 else None
            post_row = ho_df.iloc[i+1] if i != len(ho_df) - 1 else None

            # Peek the next event to avoid HO overlapping with handoverFailure (skip!!)
            if i != len(ho_df) - 1 and pd.notna(row.end) and row.end > post_row.start:
                # print('Overlapping event occurs!!')
                # print(i, row['start'], row['end'], row['type'], row['cause'])
                # print(i+1, post_row['start'], post_row['end'], post_row['type'], post_row['cause'])
                continue
            
            # Set prior event if the prior loop is skipped
            if i != 0 and pd.notna(prior_row.end) and prior_row.end > row.start:
                prior_row = ho_df.iloc[i-2] if i > 1 else None
            
            # Basic information of the current row
            tag = '_'.join([s for s in row[self.sp_columns] if pd.notna(s)])  # specific column name
            start_ts, end_ts = row['start'], row['end']  # handover start/end time
            
            # Set simple left/right bounds
            current_left_bound = start_ts + pd.Timedelta(seconds=(scope[tag][0]))
            current_right_bound = start_ts + pd.Timedelta(seconds=(scope[tag][1]))
            
            # Set left/right bounds to avoid event overlapping with each other
            if prior_row is not None:
                prior_tag = '_'.join([s for s in prior_row[self.sp_columns] if pd.notna(s)])
                prior_right_bound = prior_row['start'] + pd.Timedelta(seconds=(scope[prior_tag][1]))
                if pd.notna(prior_row['end']):
                    left_bound = min(max(current_left_bound, DrEval.interpolate(prior_right_bound, current_left_bound), prior_row['end']), start_ts)
                else:
                    left_bound = min(max(current_left_bound, DrEval.interpolate(prior_right_bound, current_left_bound), prior_row['start']), start_ts)
            else:
                left_bound = current_left_bound
            
            if post_row is not None:
                post_tag = '_'.join([s for s in post_row[self.sp_columns] if pd.notna(s)])
                post_left_bound = post_row['start'] + pd.Timedelta(seconds=(scope[post_tag][0]))
                if pd.notna(end_ts):
                    right_bound = max(min(current_right_bound, DrEval.interpolate(current_right_bound, post_left_bound), post_row['start']), end_ts)
                else:
                    right_bound = max(min(current_right_bound, DrEval.interpolate(current_right_bound, post_left_bound), post_row['start']), start_ts)
            else:
                right_bound = current_right_bound
            
            interval = P.closed(left_bound, right_bound)
            
            # Concatenate PLR from mapping list
            try:
                current_df = this_df[this_df['Timestamp'] < interval.upper].copy()
            except:
                continue
            plr_mapping = hist_model[tag].copy()
            
            current_df[f'relative_time'] = (current_df['Timestamp'] - start_ts).dt.total_seconds()
            current_df[f'window_id'] = ((current_df[f'relative_time'] + 0.005) // 0.01) * 0.01
            
            trigger_probability = prob_model[tag]
            
            if plr_mapping.empty:
                tmp = current_df.copy().rename(columns={mets: f'{mets}_x'})
            else:
                tmp = pd.merge(current_df, plr_mapping, on='window_id', how='left')
                tmp[RATE_TYPE] = tmp[RATE_TYPE].fillna(0)
                
                if not DrEval.generate_random_boolean(trigger_probability):
                    tmp[RATE_TYPE] = 0
            
            tmp['type'] = tag
            
            try:
                answer = pd.concat([answer, tmp], axis=0)
            except:
                answer = tmp.copy()
            
            # Update dataframe to accelerate the speed
            this_df = this_df[this_df[self.ts_column] >= interval.upper].copy()
        
        answer = pd.concat([answer, this_df], axis=0)
        
        # Consider stable duration
        if mets == 'lost':
            stable_df = answer[answer['tx_count'].isnull()].copy()[['packet_number', 'lost_x', 'excl', 'loex', 'Timestamp']].rename(columns={f'{mets}_x': mets})
        else:
            stable_df = answer[answer['tx_count'].isnull()].copy()[['packet_number', 'lost', 'excl_x', 'loex', 'Timestamp']].rename(columns={f'{mets}_x': mets})

        stable_df['Timestamp_sec'] = stable_df['Timestamp'].dt.floor('S')
        stable_df['relative_time'] = (stable_df['Timestamp'] - stable_df['Timestamp_sec']).dt.total_seconds() - 0.5
        stable_df['window_id'] = ((stable_df['relative_time'] + 0.01 / 2) // 0.01) * 0.01

        plr_mapping = hist_model['Stable'].copy()        
        stable_df = pd.merge(stable_df, plr_mapping, on='window_id', how='left').rename(columns={RATE_TYPE: f'{RATE_TYPE}_if_trigger'})
        trigger_prob_mapping = stable_df[~stable_df['Timestamp_sec'].duplicated()].reset_index(drop=True)[['Timestamp_sec']]
        
        trigger_probability = prob_model['Stable']
        random_bool_array = [DrEval.generate_random_boolean(trigger_probability) for _ in range(len(trigger_prob_mapping))]
        trigger_prob_mapping['trigger'] = random_bool_array

        stable_df = pd.merge(stable_df, trigger_prob_mapping, on='Timestamp_sec', how='left')
        stable_df[RATE_TYPE] = stable_df[f'{RATE_TYPE}_if_trigger'] * stable_df['trigger']
        
        stable_df['type'] = 'Stable'

        del stable_df['Timestamp_sec'], stable_df[f'{RATE_TYPE}_if_trigger'], stable_df['trigger']
        
        answer = answer[answer['tx_count'].notnull()].copy()
        
        try:
            answer = pd.concat([answer, stable_df], axis=0)
        except:
            print('******* answer *******')
            print(answer)
            print(answer.columns)
            print(answer.index.is_unique)
            print('******* stable_df *******')
            print(stable_df)
            print(stable_df.columns)
            print(stable_df.index.is_unique)
            raise
        
        answer = answer.sort_values(by='Timestamp').reset_index(drop=True)
        answer[RATE_TYPE] = answer[RATE_TYPE] / 100
        answer['Y'] = answer[RATE_TYPE].apply(DrEval.generate_random_boolean)
        
        eval_value = answer['Y'].mean() * 100
        ground_value = df[mets].mean() * 100

        answer = pd.concat([answer[['packet_number', 'Timestamp', 'type', 'relative_time', 'window_id']],
                            df[['lost', 'excl', 'loex']],
                            answer[[RATE_TYPE, 'Y']]], axis=1)
        
        return answer, eval_value, ground_value
    
    ## TODO: def anchor_by_packet

    def anchor_by_event(self, ho_df1, ho_df2, df2):
        scope, mets = self.scope_models, self.mets
        this_df = df2.copy()
        
        # 觀察 lost 有沒有出現在 (anchor_type, anchor_state, type) 的影響範圍內，若有的話，理論上取值的 df 不為空
        this_df = this_df[this_df['Y']].copy().reset_index(drop=True)
        
        # Ignore the stable state, all assumed to be no-loss
        ho_df1['anchor_type'] = 'Stable'
        ho_df1['anchor_state'] = 0

        for i, row in ho_df2.iterrows():
            prior_row = ho_df2.iloc[i-1] if i != 0 else None
            post_row = ho_df2.iloc[i+1] if i != len(ho_df2) - 1 else None

            # Peek the next event to avoid HO overlapping with handoverFailure (skip it!!)
            if i != len(ho_df2) - 1 and pd.notna(row.end) and row.end > post_row.start:
                # print('Overlapping event occurs!!')
                # print(i, row['start'], row['end'], row['type'], row['cause'])
                # print(i+1, post_row['start'], post_row['end'], post_row['type'], post_row['cause'])
                continue
            
            # Set prior event if the prior loop is skipped
            if i != 0 and pd.notna(prior_row.end) and prior_row.end > row.start:
                prior_row = ho_df2.iloc[i-2] if i > 1 else None
            
            # Basic information of the current row
            tag = '_'.join([s for s in row[self.sp_columns] if pd.notna(s)])  # specific column name
            start_ts, end_ts = row['start'], row['end']  # handover start/end time
            
            # Set simple left/right bounds
            current_left_bound = start_ts + pd.Timedelta(seconds=(scope[tag][0]))
            current_right_bound = start_ts + pd.Timedelta(seconds=(scope[tag][1]))
            
            # Set left/right bounds to avoid event overlapping with each other
            if prior_row is not None:
                prior_tag = '_'.join([s for s in prior_row[self.sp_columns] if pd.notna(s)])
                prior_right_bound = prior_row['start'] + pd.Timedelta(seconds=(scope[prior_tag][1]))
                if pd.notna(prior_row['end']):
                    left_bound = min(max(current_left_bound, DrEval.interpolate(prior_right_bound, current_left_bound), prior_row['end']), start_ts)
                else:
                    left_bound = min(max(current_left_bound, DrEval.interpolate(prior_right_bound, current_left_bound), prior_row['start']), start_ts)
            else:
                left_bound = current_left_bound
            
            if post_row is not None:
                post_tag = '_'.join([s for s in post_row[self.sp_columns] if pd.notna(s)])
                post_left_bound = post_row['start'] + pd.Timedelta(seconds=(scope[post_tag][0]))
                if pd.notna(end_ts):
                    right_bound = max(min(current_right_bound, DrEval.interpolate(current_right_bound, post_left_bound), post_row['start']), end_ts)
                else:
                    right_bound = max(min(current_right_bound, DrEval.interpolate(current_right_bound, post_left_bound), post_row['start']), start_ts)
            else:
                right_bound = current_right_bound
            
            interval = P.closed(left_bound, right_bound)

            try:
                ho_df1.loc[(ho_df1['start'] >= interval.lower) & (ho_df1['start'] < interval.upper), 'anchor_type'] = tag
                
                if not this_df[(this_df['Timestamp'] >= interval.lower) & (this_df['Timestamp'] < interval.upper)].empty:
                    ho_df1.loc[(ho_df1['start'] >= interval.lower) & (ho_df1['start'] < interval.upper), 'anchor_state'] = 1
                    
                # Update dataframe to accelerate the speed
                this_df = this_df[this_df[self.ts_column] >= interval.upper].copy()
            except:
                continue
            
        return ho_df1
    
    def hist_method_dual_by_event(self, df, ho_df):
        mets, RATE_TYPE = self.mets, self.RATE_TYPE
        scope = self.scope_models
        hist_model = self.hist_models
        dr_prob_model = self.dr_prob_models
        
        this_df = df.copy()
        for i, row in ho_df.iterrows():
            prior_row = ho_df.iloc[i-1] if i != 0 else None
            post_row = ho_df.iloc[i+1] if i != len(ho_df) - 1 else None

            # Peek the next event to avoid HO overlapping with handoverFailure (skip!!)
            if i != len(ho_df) - 1 and pd.notna(row.end) and row.end > post_row.start:
                # print('Overlapping event occurs!!')
                # print(i, row['start'], row['end'], row['type'], row['cause'])
                # print(i+1, post_row['start'], post_row['end'], post_row['type'], post_row['cause'])
                continue
            
            # Set prior event if the prior loop is skipped
            if i != 0 and pd.notna(prior_row.end) and prior_row.end > row.start:
                prior_row = ho_df.iloc[i-2] if i > 1 else None
            
            # Basic information of the current row
            tag = '_'.join([s for s in row[self.sp_columns] if pd.notna(s)])  # specific column name
            start_ts, end_ts = row['start'], row['end']  # handover start/end time
            
            # Set simple left/right bounds
            current_left_bound = start_ts + pd.Timedelta(seconds=(scope[tag][0]))
            current_right_bound = start_ts + pd.Timedelta(seconds=(scope[tag][1]))
            
            # Set left/right bounds to avoid event overlapping with each other
            if prior_row is not None:
                prior_tag = '_'.join([s for s in prior_row[self.sp_columns] if pd.notna(s)])
                prior_right_bound = prior_row['start'] + pd.Timedelta(seconds=(scope[prior_tag][1]))
                if pd.notna(prior_row['end']):
                    left_bound = min(max(current_left_bound, DrEval.interpolate(prior_right_bound, current_left_bound), prior_row['end']), start_ts)
                else:
                    left_bound = min(max(current_left_bound, DrEval.interpolate(prior_right_bound, current_left_bound), prior_row['start']), start_ts)
            else:
                left_bound = current_left_bound
            
            if post_row is not None:
                post_tag = '_'.join([s for s in post_row[self.sp_columns] if pd.notna(s)])
                post_left_bound = post_row['start'] + pd.Timedelta(seconds=(scope[post_tag][0]))
                if pd.notna(end_ts):
                    right_bound = max(min(current_right_bound, DrEval.interpolate(current_right_bound, post_left_bound), post_row['start']), end_ts)
                else:
                    right_bound = max(min(current_right_bound, DrEval.interpolate(current_right_bound, post_left_bound), post_row['start']), start_ts)
            else:
                right_bound = current_right_bound
            
            interval = P.closed(left_bound, right_bound)

            # Concatenate PLR from mapping list
            try:
                current_df = this_df[this_df['Timestamp'] < interval.upper].copy()
            except:
                continue
            plr_mapping = hist_model[tag].copy()
            
            current_df[f'relative_time'] = (current_df['Timestamp'] - start_ts).dt.total_seconds()
            current_df[f'window_id'] = ((current_df[f'relative_time'] + 0.005) // 0.01) * 0.01
            
            anchor_tag = row['anchor_type']
            anchor_state = row['anchor_state']
            
            if anchor_state == 1:
                trigger_probability = dr_prob_model[(anchor_tag, tag)][0]
            else:
                trigger_probability = dr_prob_model[(anchor_tag, tag)][1]
            
            if plr_mapping.empty:
                tmp = current_df.copy().rename(columns={mets: f'{mets}_x'})
            else:
                tmp = pd.merge(current_df, plr_mapping, on='window_id', how='left')
                tmp[RATE_TYPE] = tmp[RATE_TYPE].fillna(0)
                
                if not DrEval.generate_random_boolean(trigger_probability):
                    tmp[RATE_TYPE] = 0
            
            tmp['anchor_type'] = anchor_tag
            tmp['anchor_state'] = anchor_state
            tmp['type'] = tag

            try:
                answer = pd.concat([answer, tmp], axis=0)
            except:
                answer = tmp.copy()
            
            # Update dataframe to accelerate the speed
            this_df = this_df[this_df[self.ts_column] >= interval.upper].copy()
        
        answer = pd.concat([answer, this_df], axis=0)
        
        # Consider stable duration
        if mets == 'lost':
            stable_df = answer[answer['tx_count'].isnull()].copy()[['packet_number', 'lost_x', 'excl', 'loex', 'Timestamp']].rename(columns={f'{mets}_x': mets})
        else:
            stable_df = answer[answer['tx_count'].isnull()].copy()[['packet_number', 'lost', 'excl_x', 'loex', 'Timestamp']].rename(columns={f'{mets}_x': mets})

        stable_df['Timestamp_sec'] = stable_df['Timestamp'].dt.floor('S')
        stable_df['relative_time'] = (stable_df['Timestamp'] - stable_df['Timestamp_sec']).dt.total_seconds() - 0.5
        stable_df['window_id'] = ((stable_df['relative_time'] + 0.01 / 2) // 0.01) * 0.01

        plr_mapping = hist_model['Stable'].copy()        
        stable_df = pd.merge(stable_df, plr_mapping, on='window_id', how='left').rename(columns={RATE_TYPE: f'{RATE_TYPE}_if_trigger'})
        trigger_prob_mapping = stable_df[~stable_df['Timestamp_sec'].duplicated()].reset_index(drop=True)[['Timestamp_sec']]
        
        # Ignore the stable state, all assumed to be no-loss
        trigger_probability = dr_prob_model[('Stable', 'Stable')][0]
                
        random_bool_array = [DrEval.generate_random_boolean(trigger_probability) for _ in range(len(trigger_prob_mapping))]
        trigger_prob_mapping['trigger'] = random_bool_array

        stable_df = pd.merge(stable_df, trigger_prob_mapping, on='Timestamp_sec', how='left')
        stable_df[RATE_TYPE] = stable_df[f'{RATE_TYPE}_if_trigger'] * stable_df['trigger']
        
        stable_df['anchor_type'] = 'Stable'
        stable_df['anchor_state'] = 0
        stable_df['type'] = 'Stable'

        del stable_df['Timestamp_sec'], stable_df[f'{RATE_TYPE}_if_trigger'], stable_df['trigger']
        
        answer = answer[answer['tx_count'].notnull()].copy()
        
        try:
            answer = pd.concat([answer, stable_df], axis=0)
        except:
            print('******* answer *******')
            print(answer)
            print(answer.columns)
            print(answer.index.is_unique)
            print('******* stable_df *******')
            print(stable_df)
            print(stable_df.columns)
            print(stable_df.index.is_unique)
            raise
        
        answer = answer.sort_values(by='Timestamp').reset_index(drop=True)
        answer[RATE_TYPE] = answer[RATE_TYPE] / 100
        answer['Y'] = answer[RATE_TYPE].apply(DrEval.generate_random_boolean)
        
        eval_value = answer['Y'].mean() * 100
        ground_value = df['lost'].mean() * 100

        answer = pd.concat([answer[['packet_number', 'Timestamp', 'anchor_type', 'anchor_state', 'type', 'relative_time', 'window_id']],
                            df[['lost', 'excl', 'loex']],
                            answer[[RATE_TYPE, 'Y']]], axis=1)

        return answer, eval_value, ground_value
    
    def run_hist_method(self, N=5):
        dirc, mets = self.dirc, self.mets
        RATE_TYPE = self.RATE_TYPE
        n = len(self.filepaths)
        for i, filepath in enumerate(self.filepaths):
            
            if self.test_mode and i > 0:
                break
            
            if dirc == 'dl':
                print(f'{i+1}/{n}', filepath[0][0]); print(f'{i+1}/{n}', filepath[0][1])
                print(f'{i+1}/{n}', filepath[1][0]); print(f'{i+1}/{n}', filepath[1][1])
            else:
                print(f'{i+1}/{n}', filepath[0][0]); print(f'{i+1}/{n}', filepath[0][2])
                print(f'{i+1}/{n}', filepath[1][0]); print(f'{i+1}/{n}', filepath[1][2])
                
            if os.path.isfile(filepath[0][0]):
                ho_df1 = generate_dataframe(filepath[0][0], parse_dates=['start', 'end'])
            else:
                print('{} does not exist!!!'.format(filepath[0][0]))
                print('makefile:', filepath[0][0])
                ho_df, _ = mi_parse_handover(generate_dataframe(filepath[0][3], parse_dates=['Timestamp']))
                ho_df.to_csv(filepath[0][0], index=False)
            
            if os.path.isfile(filepath[1][0]):
                ho_df2 = generate_dataframe(filepath[1][0], parse_dates=['start', 'end'])
            else:
                print('{} does not exist!!!'.format(filepath[1][0]))
                print('makefile:', filepath[1][0])
                ho_df, _ = mi_parse_handover(generate_dataframe(filepath[1][3], parse_dates=['Timestamp']))
                ho_df.to_csv(filepath[1][0], index=False)
            
            if ho_df1.empty or ho_df2.empty:
                print('*************** EMPTY HO INFO ***************')
                continue
            
            if dirc == 'dl':
                df1 = generate_dataframe(filepath[0][1], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latest_rtt'])
                df2 = generate_dataframe(filepath[1][1], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latest_rtt'])
            else:
                df1 = generate_dataframe(filepath[0][2], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latest_rtt'])
                df2 = generate_dataframe(filepath[1][2], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latest_rtt'])
            
            df1, ho_df1, empty_data1 = data_aligner(df1, ho_df1)
            df2, ho_df2, empty_data2 = data_aligner(df2, ho_df2)
            
            if empty_data1 or empty_data2:
                print('*************** EMPTY DATA ***************')
                continue
            
            df = pd.merge(df1, df2, on='packet_number', how='inner').reset_index(drop=True)
            df1 = df[['packet_number', 'Timestamp_x', 'lost_x', 'excl_x', 'latest_rtt_x']].rename(columns={'Timestamp_x': 'Timestamp', 'lost_x': 'lost', 'excl_x': 'excl', 'latest_rtt_x': 'latest_rtt'})
            df2 = df[['packet_number', 'Timestamp_y', 'lost_y', 'excl_y', 'latest_rtt_y']].rename(columns={'Timestamp_y': 'Timestamp', 'lost_y': 'lost', 'excl_y': 'excl', 'latest_rtt_y': 'latest_rtt'})
            
            df1['excl'] = ~df1['lost'] & df1['excl']; df1['loex'] = df1['lost'] | df1['excl']
            df2['excl'] = ~df2['lost'] & df2['excl']; df2['loex'] = df2['lost'] | df2['excl']
            
            # Start processing...
            loss_rate_list = []
            answer = None
            # for iter_round in tqdm(range(N), ncols=1000):
            for iter_round in range(N):
                ans1, _, _ = self.hist_method_anchor(df1, ho_df1)
                
                if self.anchor_mode == 'by_packet':
                    avatar_df2 = self.anchor_by_packet(df2.copy(), ho_df1, ans1)
                    ans2, _, _ = self.hist_method_dual_by_packet(avatar_df2, ho_df2)
                else:
                    avatar_ho_df2 = self.anchor_by_event(ho_df2.copy(), ho_df1, ans1)
                    ans2, _, _ = self.hist_method_dual_by_event(df2, avatar_ho_df2)
                
                ans = pd.merge(ans1, ans2, on='packet_number', how='inner').reset_index(drop=True)
                ans[mets] = (ans[f'{mets}_x']) & (ans[f'{mets}_y'])
                ans['Y'] = (ans['Y_x']) & (ans['Y_y'])
                ans = ans[['packet_number', 'Timestamp_x', 'Timestamp_y', 'type_x', 'anchor_type', 'anchor_state', 'type_y',
                           'relative_time_x', 'window_id_x', 'relative_time_y', 'window_id_y',
                           f'{mets}_x', f'{mets}_y', mets, f'{RATE_TYPE}_x', 'Y_x', f'{RATE_TYPE}_y', 'Y_y', 'Y']]
                
                handle = (ans1['Y']) & (ans2['Y'])
                eval_value = handle.mean() * 100
                # print('eval:', eval_value, len(handle))
                
                handle = (df1[mets]) & (df2[mets])
                ground_value = handle.mean() * 100
                # print('ground:', ground_value, len(handle))
                
                if answer is None:
                    answer = ans.copy()
                    answer = answer.rename(columns={f'{RATE_TYPE}_x': f'{RATE_TYPE}_x_0', 'Y_x': 'Y_x_0',
                                                    f'{RATE_TYPE}_y': f'{RATE_TYPE}_y_0', 'Y_y': 'Y_y_0',
                                                    'Y': 'Y_0'})
                else:
                    answer = pd.concat([answer, ans[[f'{RATE_TYPE}_x', 'Y_x', f'{RATE_TYPE}_y', 'Y_y', 'Y']]], axis=1)
                    answer = answer.rename(columns={f'{RATE_TYPE}_x': f'{RATE_TYPE}_x_{iter_round}', 'Y_x': f'Y_x_{iter_round}',
                                                    f'{RATE_TYPE}_y': f'{RATE_TYPE}_y_{iter_round}', 'Y_y': f'Y_y_{iter_round}',
                                                    'Y': f'Y_{iter_round}'})
                
                loss_rate_list.append(eval_value)
            
            def remove_min_max(nums, epsilon=1e-9):
                if len(nums) < 5:
                    return nums
                min_val = min(nums)
                max_val = max(nums)
                nums = [num for num in nums if abs(num - min_val) > epsilon and abs(num - max_val) > epsilon]
                return nums

            loss_rate_list_ = remove_min_max(loss_rate_list)
            mean_value = np.mean(loss_rate_list_)
            std_deviation = np.std(loss_rate_list_)
            error = mean_value - ground_value
            
            path1 = filepath[0][1] if dirc == 'dl' else filepath[0][2]
            path2 = filepath[1][1] if dirc == 'dl' else filepath[1][2]            
            
            def find_sm_label(path):
                sm_index = path.index("sm")  # 找到 "sm" 的位置
                next_slash_index = path.index("/", sm_index)  # 从 "sm" 的位置开始找到下一个斜杠 "/"
                sm_dev = path[sm_index:next_slash_index]  # 截取 "sm00" 标签
                return sm_dev

            def find_trip_label(path):
                sm_index = path.index("#")  # 找到 "#" 的位置
                next_slash_index = path.index("/", sm_index)  # 从 "#" 的位置开始找到下一个斜杠 "/"
                sm_trip = path[sm_index:next_slash_index]  # 截取 "#01" 标签
                return sm_trip
            
            sm_dev = f'{find_sm_label(path1)}+{find_sm_label(path2)}'
            sm_trip = f'{find_trip_label(path1)}+{find_trip_label(path2)}'
            self.records.append((loss_rate_list, mean_value, std_deviation, ground_value, error, sm_dev, sm_trip, path1, path2))
            
            # Save Answers
            # if self.save_answer:
            #     # save_path = os.path.join(self.path2results, self.sr_model_name, self.dr_model_name, self.dirc_mets, f'{self.save_name}_iter{N}')
            #     save_path = os.path.join(self.path2results, self.sr_model_name, self.dataset_type, self.dr_model_name, self.dirc_mets, f'{self.save_name}_iter{N}')
            #     if not os.path.isdir(save_path):
            #         os.makedirs(save_path)
                    
            #     save_path = os.path.join(save_path, path1.replace('/', '\\')[:-4]+path2.replace('/', '\\'))
            #     print(save_path)
                
            #     answer.to_csv(save_path, index=False)
            
            # Save Results
            save_path = self.save_path
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_path = os.path.join(save_path, f'{self.save_name}.pkl')
            print(save_path)
            
            with open(save_path, 'wb') as f:
                pickle.dump(self.records, f)
            
            # Update plot iteration number
            self.iter_num = N
    
    def plot(self, title=None):
        RATE_TYPE = self.RATE_TYPE
        
        if title is None:
            if self.model_corr == 'mle':
                title = f'DR {self.DIRC_TYPE} {self.RATE_TYPE} | {self.model_name}_{self.route} (MLE)'
            elif self.model_corr == 'adjust':
                suffix = self.model_corr.capitalize()
                title = f'DR {self.DIRC_TYPE} {self.RATE_TYPE} | {self.model_name}_{self.route} ({suffix})'
            else:
                mode = self.model_corr.split('_')[0]
                suffix = mode.capitalize() + ' ' + 'C.C.'
                title = f'DR {self.DIRC_TYPE} {self.RATE_TYPE} | {self.model_name}_{self.route} ({suffix})'
        
        # Sample data
        print("========== SELF RECORDS ==========")
        for s in self.records:
            print(s)
        x = [s[3] for s in self.records]  # Ground truths
        y = [s[1] for s in self.records]  # Mean values for evaluation
        y_error = [s[2] for s in self.records]  # Standard deviations for error bars
        
        tau, p_value = kendalltau(x, y)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(6, 4))

        # Scatter plot with error bars and horizontal caps
        # ax.errorbar(x, y, yerr=y_error, linestyle='None', marker='o', color='tab:blue', capsize=5)
        ax.scatter(x, y, linestyle='None', marker='o', color='tab:blue', label='Experiment Data')

        # Annotate RMSE From the ground truths
        rmse = np.sqrt(mean_squared_error(x, y))
        rmse_rate = rmse / np.mean(x) * 100
        slope_annotation = f'RMSE: {rmse:.3f} ({rmse_rate:.1f} %)'
        ax.annotate(slope_annotation, xy=(0.45, 0.85), xycoords='axes fraction', fontsize=13, fontstyle='italic', fontweight='bold', color='black')
        slope_annotation = f'τ (p.v.): {tau:.2f} ({p_value:.3f})'
        ax.annotate(slope_annotation, xy=(0.45, 0.80), xycoords='axes fraction', fontsize=13, fontstyle='italic', fontweight='bold', color='black')
        
        underestimations = [1 for true, pred in zip(x, y) if pred < true]
        underestimation_rate = sum(underestimations) / len(x) * 100
        
        overestimations = [1 for true, pred in zip(x, y) if pred > true]
        overestimation_rate = sum(overestimations) / len(x) * 100
        
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        
        ax.set_xlim(0, max(x_limits[1], y_limits[1]))
        ax.set_ylim(0, max(x_limits[1], y_limits[1]))

        # 45-Degree Line
        ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='-', linewidth=1.1, color='tab:pink', label='45-Degree Line', alpha=0.9)
        
        # 创建 x 值范围
        x_limits = ax.get_xlim()  # 获取 x 的边界
        x_values = np.linspace(x_limits[0], x_limits[1], 100)  # 使用 x 的边界值作为范围

        # Calculate the value of y = 1.1x & y = 0.9x
        y_upper = 1.1 * x_values
        y_lower = 0.9 * x_values

        # Set labels and title
        ax.set_xlabel(f'{RATE_TYPE} Ground Truth')
        ax.set_ylabel(f'{RATE_TYPE} Evaluated')
        if title is not None:
            ax.set_title(title)
        # Add a legend
        ax.legend()
        fig.set_size_inches(6, 6)  # square figure
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        
        # Save figure
        save_path = self.save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, f'{self.save_name}.png')
        
        print(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show plot
        # plt.show()
        plt.close(fig)
        

        save_path = self.save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, f'{self.save_name}.csv')
        
        print(save_path)
        with open(save_path, mode='w') as file:
            writer = csv.writer(file)
            # 寫入表頭
            writer.writerow(['RMSE (%)', 'τ (p.v.)', 'Underestimation (%)', 'Overestimation (%)'])
            # 寫入資料
            writer.writerow([f'{rmse:.3f} ({rmse_rate:.1f}%)', f'{tau:.2f} ({p_value:.3f})', f'{underestimation_rate:.1f}%', f'{overestimation_rate:.1f}%'])
        