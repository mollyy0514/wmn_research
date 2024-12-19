import os
import pickle
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import portion as P
from tqdm import tqdm
from scipy.integrate import quad
from scipy.stats import gaussian_kde
from allutils import *

class SrProfile():
    def __init__(
        self, filepaths, model_prefix='Test',
        model_id=None, model_dscp=None, save_path='.',
        epochs=2, dirc_mets='dl_lost',
        scope=None, sp_columns=['type'], ts_column='Timestamp',
        w_size=0.01, sd_factor=3,
        test_mode=False
    ):
        if model_id is None:
            raise TypeError("請輸入模型編號")
        
        self.filepaths = copy.deepcopy(filepaths)
        self.model_name = model_id if model_dscp is None else model_id + '_' + model_dscp
        self.model_prefix = model_prefix
        self.save_path = save_path
        
        self.dirc_mets = dirc_mets
        self.dirc, self.mets = dirc_mets[:2], dirc_mets[-4:]
        self.DIRC_TYPE = 'Downlink' if self.dirc == 'dl' else 'Uplink'
        self.METS_TYPE = 'Packet Loss Rate (%)' if self.mets == 'lost' else 'Excessive Latency Rate (%)' if self.mets == 'excl' else 'Average Latency (ms)'
        self.RATE_TYPE = 'PLR' if self.mets == 'lost' else 'ELR' if self.mets == 'excl' else 'Latency'
        self.sp_columns = sp_columns[:] # specific column name
        self.ts_column = ts_column
        self.w_size = w_size
        self.sd_factor = sd_factor
        self.test_mode = test_mode

        print(self.save_path, self.model_name, self.model_prefix, self.dirc_mets)

        if scope is None:
            self.scope = {
                **{key: (-5.0, 5.0) for key in ['LTEH', 'ENBH', 'MCGH', 'MNBH', 'SCGM', 'SCGA', 'SCGR-I', 'SCGR-II', 'SCGC-I', 'SCGC-II']},
                **{key: (-10.0, 10.0) for key in ['SCGF', 'MCGF', 'NASR']}, 
                'Stable': (-1.0, 1.0)
            }
        else:
            self.scope = copy.deepcopy(scope)
        
        self.Container = None
        self.Profile = None
        
        self.scope_models = None
        self.hist_models = None
        self.kde_models = None
        self.plr_models = None
        self.prob_models = None

        # Building Model
        epoch_mapping = {0: '1st', 1: '2nd', 2: '3rd', 3: '4th', 4: '5th'}
        for epoch in tqdm(range(epochs)):
            
            if epoch == epochs - 1:
                self.epoch = 'last'
            else:
                self.epoch = epoch_mapping[epoch]
                
            if epoch > 0:
                self.scope = copy.deepcopy(self.scope_models['result'])
            
            self.reset()
            self.construct_profile()
            self.modeling()
            print('--------------')
            print('Size of Profiles:', getsizeof(self.Profile))
            print('Total Size:', getsizeof(self))
            
            if self.epoch == 'last':
                self.plot()
        
        self.save_models()

    def reset(self):
        self.Container = { tag: { 'dist_table': [],
                                  'relative_loex_timestamp': [],
                                  'relative_timestamp': [],
                                  'interruption_time': [],
                                  'trigger_loex': [],
                                  'event_count': [] }
                            for tag in self.scope.keys() }
        
        self.Profile = { tag: { 'dist_table': None,
                                'relative_loex_timestamp': [],
                                'relative_timestamp': [],
                                'interruption_time': [],
                                'trigger_loex': [],
                                'event_count': 0 }
                            for tag in self.scope.keys() }
        
        self.scope_models = { stage: copy.deepcopy(self.scope) for stage in ['initial', 'result'] }
        self.hist_models = { tag: None for tag in self.scope.keys() }
        self.kde_models = { tag: (None, None, None) for tag in self.scope.keys() }
        self.plr_models = { tag: 0.0 for tag in self.scope.keys() }
        self.prob_models = { tag: 0.5 for tag in self.scope.keys() }

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
    
    def dist_aggregate(self, tables):
        mets, RATE_TYPE = self.mets, self.RATE_TYPE
        
        table = pd.DataFrame(columns=['window_id', 'tx_count', mets, 'latency','avg_cwnd_change', 'good_tx_count', 'latest_rtt'])
        table[mets] = table[mets].astype('Int32')
        table['window_id'] = table['window_id'].astype('float32')
        table['tx_count'] = table['tx_count'].astype('Int32')
        table['avg_cwnd_change'] = table['avg_cwnd_change'].astype('float32')
        table['latency'] = table['latency'].astype('float32')
        table['good_tx_count'] = table['good_tx_count'].astype('Int32')
        table['latest_rtt'] = table['latest_rtt'].astype('Int32')
        
        tables = [t for t in tables if t is not None]
        for this_table in tables:
            table = table.merge(this_table, on=['window_id'], how='outer').fillna(0)
            table['tx_count'] = table['tx_count_x'] + table['tx_count_y']
            table[mets] = table[f'{mets}_x'] + table[f'{mets}_y']
            table['avg_cwnd_change'] = table[['avg_cwnd_change_x', 'avg_cwnd_change_y']].mean(axis=1, skipna=True)
            table['latency'] = table[['latency_x', 'latency_y']].mean(axis=1, skipna=True)
            table['good_tx_count'] = table['good_tx_count_x'] + table['good_tx_count_y']
            table['latest_rtt'] = table[['latest_rtt_x', 'latest_rtt_y']].mean(axis=1, skipna=True)
            table = table[['window_id', 'tx_count', mets, 'latency', 'avg_cwnd_change', 'good_tx_count', 'latest_rtt']]
        
        table[RATE_TYPE] = table[mets] / (table['tx_count'] + 1e-9) * 100
        table[RATE_TYPE] = table[RATE_TYPE].astype('float32')
        
        table = table[['window_id', 'tx_count', mets, 'latency', 'avg_cwnd_change', 'good_tx_count', 'latest_rtt', RATE_TYPE]].sort_values(by=['window_id']).reset_index(drop=True)
        return table
    
    @staticmethod
    def downsample(data, sample_size=100000):
        return mean_downsample(data, sample_size=sample_size)
    
    @staticmethod
    def total_area_histogram_with_centers(x_centers, heights, bin_width):
        # 計算每個 bin 的面積並相加
        total_area = bin_width * sum(heights)
        return total_area
    
    # @staticmethod
    # def total_area_kde(kde, lower_bound=-np.inf, upper_bound=np.inf):
    #     # 定義積分函數
    #     def integrand(x):
    #         return kde(x)
    #     total_area, _ = quad(integrand, lower_bound, upper_bound)
    #     return total_area

    def create_instance(self, df, center, interval):
        mets, w_size = self.mets, self.w_size
        df = df[(df[self.ts_column] >= interval.lower) & (df[self.ts_column] < interval.upper)].copy().reset_index(drop=True)
        
        # Relative window converted from timestamp
        df['relative_time'] = (df['Timestamp'] - center).dt.total_seconds() # relative window time
        df['window_id'] = ((df['relative_time'] + w_size / 2) // w_size) * w_size  # 四捨五入
        # Calculate the change in congestion window
        df['filled_cwnd'] = df['congestion_window'].fillna(method='ffill')
        first_valid_index = df['filled_cwnd'].first_valid_index()
        if first_valid_index is not None:
            first_filled_cwnd = df.loc[first_valid_index, 'filled_cwnd']
            df['avg_cwnd_change'] = round((df['filled_cwnd'] - first_filled_cwnd) * 100 / first_filled_cwnd, 4)
        else:
            df['avg_cwnd_change'] = None        # Restore NaN values in the 'avg_cwnd_change' where the original 'congestion_window' was NaN
        df['avg_cwnd_change'] = df['avg_cwnd_change'].where(df['congestion_window'].notna())
        df.drop(columns=['filled_cwnd'], inplace=True)
        
        if mets == 'lost':
            loex_df = df[df['lost']].copy()
            ts_group = df.groupby(['window_id'])
            table = ts_group.agg({'lost': ['count', 'sum'], 
                                  'latency': ['mean'], 
                                  'Timestamp': ['first'], 
                                  'avg_cwnd_change': ['mean'],
                                  'retransmit': ['sum'],
                                  'latest_rtt': ['mean']
                                  }).reset_index()
        elif mets == 'excl':
            # only calculate the packets with excl, but didn't used???
            df['excl_exact'] = ~df['lost'] & df['excl']
            loex_df = df[df['excl_exact']].copy()
            ts_group = df.groupby(['window_id'])
            table = ts_group.agg({'excl_exact': ['count', 'sum'], 
                                  'latency': ['mean'], 
                                  'Timestamp': ['first'], 
                                  'avg_cwnd_change': ['mean'],
                                  'retransmit': ['sum'],
                                  'latest_rtt': ['mean']
                                  }).reset_index()
        else:
            loex_df = df[~df['lost']].copy()
            ts_group = df.groupby(['window_id'])
            table = ts_group.agg({'latency': ['count', 'mean'], 
                                  'latency': ['mean'], 
                                  'Timestamp': ['first'], 
                                  'retransmit': ['sum'],
                                  'latest_rtt': ['mean']
                                  }).reset_index()
            
        table.columns = ['window_id', 'tx_count', mets, 'latency', 'Timestamp', 'avg_cwnd_change', 'good_tx_count', 'latest_rtt']
        
        return table, loex_df['relative_time'].to_list(), df['relative_time'].to_list()

    def setup_profile(self, df, ho_df):
        scope, mets = self.scope, self.mets
        
        # Initialize "Register", which record a hnadover event / stable info (e.g. pkl) 
        Register = { tag: { 'dist_table': [],
                            'relative_loex_timestamp': [],
                            'relative_timestamp': [],
                            'interruption_time': [],
                            'trigger_loex': [] }
                    for tag in scope.keys() }
        this_df = df.copy()

        for i, row in ho_df.iterrows():
            prior_row = ho_df.iloc[i-1] if i != 0 else None
            post_row = ho_df.iloc[i+1] if i != len(ho_df) - 1 else None
            
            # Peek the next event to avoid HO overlapping with handoverFailure (skip it!!)
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
            interruption_time = (end_ts - start_ts).total_seconds() if pd.notna(end_ts) else 0  # handover interruption time
            
            # Set simple left/right bounds
            current_left_bound = start_ts + pd.Timedelta(seconds=(scope[tag][0]))
            current_right_bound = start_ts + pd.Timedelta(seconds=(scope[tag][1]))
            
            # Set left/right bounds to avoid event overlapping with each other
            if prior_row is not None:
                prior_tag = '_'.join([s for s in prior_row[self.sp_columns] if pd.notna(s)])
                prior_right_bound = prior_row['start'] + pd.Timedelta(seconds=(scope[prior_tag][1]))
                if pd.notna(prior_row['end']):
                    left_bound = min(max(current_left_bound, SrProfile.interpolate(prior_right_bound, current_left_bound), prior_row['end']), start_ts)
                else:
                    left_bound = min(max(current_left_bound, SrProfile.interpolate(prior_right_bound, current_left_bound), prior_row['start']), start_ts)
            else:
                left_bound = current_left_bound
            
            if post_row is not None:
                post_tag = '_'.join([s for s in post_row[self.sp_columns] if pd.notna(s)])
                post_left_bound = post_row['start'] + pd.Timedelta(seconds=(scope[post_tag][0]))
                if pd.notna(end_ts):
                    right_bound = max(min(current_right_bound, SrProfile.interpolate(current_right_bound, post_left_bound), post_row['start']), end_ts)
                else:
                    right_bound = max(min(current_right_bound, SrProfile.interpolate(current_right_bound, post_left_bound), post_row['start']), start_ts)
            else:
                right_bound = current_right_bound
            
            interval = P.closed(left_bound, right_bound)

            # Consider the "stable" duration before an event starts
            try:
                stable_df = this_df[this_df[self.ts_column] < interval.lower].copy()
            except:
                continue
            stable_df['Timestamp_to_sec'] = stable_df['Timestamp'].dt.floor('S')
            if not stable_df.empty:
                unique_timestamps = stable_df['Timestamp_to_sec'].unique()
                tmp_df = stable_df.copy()
                for ts in unique_timestamps:
                    stable_center = ts + pd.Timedelta(seconds=0.5)
                    stable_interval = P.closed(ts, min(ts + pd.Timedelta(seconds=1), interval.lower))
                    
                    # Create an instance of stable profile
                    dist_table, relative_loex_timestamp, relative_timestamp = self.create_instance(tmp_df.copy(), stable_center, stable_interval)
                    
                    # Feed into "Register"
                    if len(relative_loex_timestamp):
                        Register['Stable']['trigger_loex'].append(1)
                        Register['Stable']['dist_table'].append(dist_table)
                        Register['Stable']['relative_loex_timestamp'] += relative_loex_timestamp
                    else:
                        Register['Stable']['trigger_loex'].append(0)
                    Register['Stable']['interruption_time'].append((stable_interval.upper - stable_interval.lower).total_seconds())
                    Register['Stable']['relative_timestamp'] += relative_timestamp
                    
                    # Update dataframe to accelerate
                    tmp_df = tmp_df[tmp_df[self.ts_column] >= ts + pd.Timedelta(seconds=1)]
            
            # Create an instance of "handover" profile
            dist_table, relative_loex_timestamp, relative_timestamp = self.create_instance(this_df.copy(), start_ts, interval)
            
            # Feed into "Register"
            if len(relative_loex_timestamp):
                Register[tag]['trigger_loex'].append(1)
                Register[tag]['dist_table'].append(dist_table)
                Register[tag]['relative_loex_timestamp'] += relative_loex_timestamp
            else:
                Register[tag]['trigger_loex'].append(0)
            Register[tag]['interruption_time'].append(interruption_time)
            Register[tag]['relative_timestamp'] += relative_timestamp
            
            # Update dataframe to accelerate the speed
            this_df = this_df[this_df[self.ts_column] >= interval.upper].copy()

        # Consider the stable duration after the last event ends
        stable_df = this_df.copy()
        stable_df['Timestamp_to_sec'] = stable_df['Timestamp'].dt.floor('S')

        if not stable_df.empty:
            unique_timestamps = stable_df['Timestamp_to_sec'].unique()
            
            tmp_df = stable_df.copy()
            for ts in unique_timestamps:
                stable_center = ts + pd.Timedelta(seconds=0.5)
                stable_interval = P.closed(ts, ts + pd.Timedelta(seconds=1))
                
                # Create an instance of stable profile
                dist_table, relative_loex_timestamp, relative_timestamp = self.create_instance(tmp_df.copy(), stable_center, stable_interval)
                
                # Feed into "Register"
                if len(relative_loex_timestamp):
                    Register['Stable']['trigger_loex'].append(1)
                    Register['Stable']['dist_table'].append(dist_table)
                    Register['Stable']['relative_loex_timestamp'] += relative_loex_timestamp
                else:
                    Register['Stable']['trigger_loex'].append(0)
                Register['Stable']['interruption_time'].append((stable_interval.upper - stable_interval.lower).total_seconds())
                Register['Stable']['relative_timestamp'] += relative_timestamp
                
                # Update dataframe to accelerate
                tmp_df = tmp_df[tmp_df[self.ts_column] >= ts + pd.Timedelta(seconds=1)]

        return Register
    
    def construct_profile(self):
        scope, dirc, mets = self.scope, self.dirc, self.mets
        n = len(self.filepaths)
        for i, filepath in enumerate(self.filepaths):
            
            if self.test_mode and i > 0:
                break

            path = filepath[1] if dirc == 'dl' else filepath[2]
            print(f'{i+1}/{n}', filepath[0]); print(f'{i+1}/{n}', path)
            
            # generate hondover dataframe
            if os.path.isfile(filepath[0]):
                ho_df = generate_dataframe(filepath[0], parse_dates=['start', 'end'])
            else:
                print('{} does not exist!!!'.format(filepath[0]))
                print('makefile:', filepath[0])
                ho_df, _ = mi_parse_handover(generate_dataframe(filepath[3], parse_dates=['Timestamp']))
                ho_df.to_csv(filepath[0], index=False)
            
            if ho_df.empty:
                print('*************** EMPTY HO INFO ***************')
                continue
            
            # Generate dataframes for ul or dl sent files
            if dirc == 'dl':
                # df = generate_dataframe(filepath[1], parse_dates=['Timestamp'], usecols=['seq', 'Timestamp', 'lost', 'excl', 'latency'])
                df = generate_dataframe(filepath[1], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latency', 'latest_rtt', 'congestion_window', 'retransmit'])
            else:
                # df = generate_dataframe(filepath[2], parse_dates=['Timestamp'], usecols=['seq', 'Timestamp', 'lost', 'excl', 'latency'])
                df = generate_dataframe(filepath[2], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latency', 'latest_rtt', 'congestion_window', 'retransmit'])
            df, ho_df, empty_data = data_aligner(df, ho_df)
            
            if empty_data:
                print('*************** EMPTY DATA ***************')
                continue

            # Append "Register" for each trace to setup "Container"
            Register = self.setup_profile(df, ho_df)
            # print(Register)
            for tag in scope.keys():
                table = self.dist_aggregate(Register[tag]['dist_table'])
                self.Container[tag]['dist_table'].append(table)
                self.Container[tag]['relative_loex_timestamp'].append(Register[tag]['relative_loex_timestamp'])
                self.Container[tag]['relative_timestamp'].append(Register[tag]['relative_timestamp'])
                self.Container[tag]['trigger_loex'].append(Register[tag]['trigger_loex'])
                self.Container[tag]['interruption_time'].append(Register[tag]['interruption_time'])
                self.Container[tag]['event_count'].append(len(Register[tag]['interruption_time']))
            Register = {} # Reset "Register" after "Container" setup

        # Append "Container" data into "Profile", and reset "Container" after storing int "Profile"
        for tag in scope.keys():
            self.Profile[tag]['dist_table'] = self.dist_aggregate(self.Container[tag]['dist_table'])
            self.Container[tag]['dist_table'] = []
            
            data = []
            for lst in self.Container[tag]['relative_loex_timestamp']:
                data += lst
            self.Profile[tag]['relative_loex_timestamp'] = SrProfile.downsample(data)
            self.Container[tag]['relative_loex_timestamp'] = []
            
            data = []
            for lst in self.Container[tag]['relative_timestamp']:
                data += lst
            self.Profile[tag]['relative_timestamp'] = SrProfile.downsample(data)
            self.Container[tag]['relative_timestamp'] = []
            
            del data
            
            for lst in self.Container[tag]['trigger_loex']:
                self.Profile[tag]['trigger_loex'] += lst
            self.Container[tag]['trigger_loex'] = []
            
            for lst in self.Container[tag]['interruption_time']:
                self.Profile[tag]['interruption_time'] += lst
            self.Container[tag]['interruption_time'] = []
            
            self.Profile[tag]['event_count'] += sum(self.Container[tag]['event_count'])
            self.Container[tag]['event_count'] = []

    def modeling(self):
        scope, sd_factor, w_size = self.scope, self.sd_factor, self.w_size
        dirc, mets, RATE_TYPE = self.dirc, self.mets, self.RATE_TYPE
        
        for tag in scope.keys():
            # print('===============================================================')
            # print(tag, self.dirc_mets)
            
            left_bound, right_bound = scope[tag]
            table = self.Profile[tag]['dist_table']
            loex_data = self.Profile[tag]['relative_loex_timestamp']
            xmit_data = self.Profile[tag]['relative_timestamp']
            trigger_lst = self.Profile[tag]['trigger_loex']
            
            self.hist_models[tag] = table.copy()
            
            if len(trigger_lst) == 0:
                continue
            
            estimated_p = sum(trigger_lst) / len(trigger_lst)
            self.prob_models[tag] = estimated_p
            
            PLR = sum(table[mets]) / (sum(table['tx_count']) + 1e-9) * 100
            self.plr_models[tag] = PLR
            
            # if loex_data == 1, then kde function bumps error: 
            # ValueError: `dataset` input should have multiple elements.
            if len(loex_data) < 2:
                continue
            
            if self.epoch == 'last':
                left_bound, right_bound = scope[tag]
            else:
                if tag == 'Stable':
                    mean = 0
                    left_bound, right_bound = -0.5, 0.5
                else:
                    loex_table = table[table[mets] > 0].reset_index(drop=True)
                    mean, stdev = np.mean(loex_data), np.std(loex_data)
                    left_bound = math.floor(max(left_bound, mean - sd_factor * stdev, loex_table.iloc[0]['window_id']) * 10) / 10
                    right_bound = math.ceil(min(right_bound, mean + sd_factor * stdev, loex_table.iloc[-1]['window_id']) * 10) / 10
                
                self.scope_models['result'][tag] = (left_bound, right_bound)
            
            x = np.asarray(table['window_id'], dtype=np.float64)
            y = np.asarray(table[RATE_TYPE], dtype=np.float64)
            
            # 計算直方圖的面積
            hist_area = SrProfile.total_area_histogram_with_centers(x, y, w_size)
            # print("Total area of histogram:", hist_area)
            
            # kde1 = gaussian_kde(loex_data)
            # kde2 = gaussian_kde(xmit_data)
            # def kde(x):
            #     kde2_values = kde2(x)
            #     # 檢查 kde2 是否為零，如果是則返回一個超大值，把 loss rate 壓成 0
            #     kde2_values[kde2_values == 0] = 1e9
            #     return kde1(x) / kde2_values
            
            # # 計算 KDE 下的總面積（只計算正負3個標準差內的點，理論上 scalar 會稍微高估，但不會太多）
            # kde_area = SrProfile.total_area_kde(kde, left_bound, right_bound)
            # # print("Total area under KDE:", kde_area)
            
            # scalar = hist_area / kde_area
            # # print("Scalar:", scalar)
            
            # self.kde_models[tag] = (scalar, kde1, kde2)
    def plot(self):
        # 把 ax_twin 註解掉就不會有 number of packets transmitted (tx_count) 的 y-label
        scope, dirc, mets = self.scope, self.dirc, self.mets
        sd_factor = self.sd_factor
        METS_TYPE, RATE_TYPE, DIRC_TYPE = self.METS_TYPE, self.RATE_TYPE, self.DIRC_TYPE
        
        for tag in scope.keys():
            print("NOW: ", tag)
            loex_data = self.Profile[tag]['relative_loex_timestamp']
            xmit_data = self.Profile[tag]['relative_timestamp']
            
            left_bound, right_bound = self.scope_models['result'][tag]
            table = self.hist_models[tag]
            scalar, kde1, kde2 = self.kde_models[tag]
            
            if len(loex_data) == 0:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            x = np.asarray(table['window_id'], dtype=np.float64)
            y1 = np.asarray(table['tx_count'], dtype=np.float64)
            y2 = np.asarray(table[RATE_TYPE], dtype=np.float64)
            y3 = np.asarray(table['avg_cwnd_change'], dtype=np.float64)
            y4 = np.asarray(table['latency'], dtype=np.float64)

            throughput = np.where(y1 != 0, y1 * 1223 * 8 * 10 / (1e6 * sum(self.Profile[tag]['trigger_loex'])), 0)  # Handle zero tx_count to avoid division by zero
            
            ax_twin = ax.twinx()
            # Change tx_packet to throughput
            # ax_twin.bar(x, y1, label='tx_packet', color='tab:blue', width=0.01, alpha=0.15)
            ax_twin.bar(x, throughput, label='Throughput (Mbps)', color='tab:blue', width=0.01, alpha=0.15)
            ax_twin.set_ylabel('Throughput (Mbps)', color='tab:blue')  # Set the new y-label for throughput
            ax_twin.tick_params(axis='y', labelcolor='tab:blue')

            ax.bar(x, y2, label='loss_rate', color='tab:blue', width=0.01, alpha=0.97)

            # Plot avg_cwnd_change on the right y-axis
            ax3 = ax.twinx()  # Create the y-axis on the right side
            ax3.spines['right'].set_position(('outward', 60))  # Move the axis to avoid overlap
            ax3.plot(x, y3, label='avg_cwnd_change', color='tab:green', linestyle='-', linewidth=1)
            ax3.set_ylabel('Avg CWND Change (%)', color='tab:green')
            ax3.tick_params(axis='y', labelcolor='tab:green')
            # Make the y-limits symmetrical for avg_cwnd_change
            max_abs_value = max(np.abs(y3))
            ax3.set_ylim(-max_abs_value, max_abs_value)  # Set y-limits to be symmetrical
            # Mark the max and min avg_cwnd_change without displaying values
            max_cwnd_idx = np.argmax(y3)
            min_cwnd_idx = np.argmin(y3)
            # Mark max & min cwnd_change with a green dot
            # 'ko': Marks the points with green dots ('k' for black, 'o' for circle marker).
            ax3.plot(x[max_cwnd_idx], y3[max_cwnd_idx], 'ko', label='Max CWND', markersize=3)
            ax3.plot(x[min_cwnd_idx], y3[min_cwnd_idx], 'ko', label='Min CWND', markersize=3)
            # Add 'Max' label
            ax3.annotate('MAX', xy=(x[max_cwnd_idx], y3[max_cwnd_idx]), 
                        xytext=(x[max_cwnd_idx] + 0.05, y3[max_cwnd_idx] + 0.1),
                        fontsize=9, fontweight='bold', color='black')
            # Add 'Min' label
            ax3.annotate('MIN', xy=(x[min_cwnd_idx], y3[min_cwnd_idx]), 
                        xytext=(x[min_cwnd_idx] + 0.05, y3[min_cwnd_idx] - 0.1),
                        fontsize=9, fontweight='bold', color='black')
            
            # Plot latency on another right y-axis
            ax4 = ax.twinx()  # Create another y-axis on the right side
            ax4.spines['right'].set_position(('outward', 120))  # Further move the axis to avoid overlap
            ax4.plot(x, y4, label='latency', color='tab:pink', linestyle='--', linewidth=0.6)
            ax4.set_ylabel('Latency (ms)', color='tab:pink')
            ax4.tick_params(axis='y', labelcolor='tab:pink')
            # # Label the largest latency value on the plot
            # max_latency_idx = np.argmax(y4)  # Find index of the largest latency
            # max_latency_value = y4[max_latency_idx]
            # max_latency_x = x[max_latency_idx]
            # ax4.annotate(f'{max_latency_value:.2f} ms', xy=(max_latency_x, max_latency_value), 
            #             xytext=(max_latency_x + 0.1, max_latency_value + 0.1),
            #             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5),
            #             fontsize=10, fontweight='bold', color='red')

            # if kde1 is not None and kde2 is not None:
            #     x = np.linspace(min(xmit_data), max(xmit_data), 1000)
                
            #     def kde(x):
            #         kde2_values = kde2(x)
            #         # 檢查 kde2 是否為零，如果是則返回一個超大值，把 loss/excl rate 壓成 0
            #         kde2_values[kde2_values == 0] = 1e9
            #         return kde1(x) / kde2_values
            
            #     density = scalar * kde(x)
            #     ax.fill_between(x, density, label='KDE', color='tab:orange', alpha=0.45, linewidth=0)

            # find the scope and boundaries
            ax.axvline(x=0, color='red', linestyle='-', alpha=0.5)
            ax.axvline(x=left_bound, color='blue', linestyle='--', label=f'-{sd_factor} Std')
            ax.axvline(x=right_bound, color='blue', linestyle='--', label=f'+{sd_factor} Std')
            
            bottom, top = ax.get_ylim()
            ax.text(left_bound, bottom-0.05*(top-bottom), '{:.1f}'.format(left_bound), ha='center', fontweight='bold', fontsize=10, color='blue')
            ax.text(right_bound, bottom-0.05*(top-bottom), '{:.1f}'.format(right_bound), ha='center', fontweight='bold', fontsize=10, color='blue')
            
            left, right = ax.get_xlim()
            count = self.Profile[tag]['event_count']
            intr = round(np.mean(self.Profile[tag]['interruption_time']), 3)
            trigger = sum(self.Profile[tag]['trigger_loex'])
            trigger_rate = round(self.prob_models[tag] * 100, 1)
            LossR = round(sum(table[mets]) / (sum(table['tx_count']) + 1e-9) * 100, 2)
            LossR_ = round(LossR * trigger_rate / 100, 2)
            max_cwnd = np.asarray(table['avg_cwnd_change']).max()
            min_cwnd = np.asarray(table['avg_cwnd_change']).min()
            cwnd_diff = round(max_cwnd-min_cwnd, 2)
            avg_latency = round(np.asarray(table['latency']).mean(), 2)
            goodput = round((sum(table['good_tx_count']) / sum(table['tx_count'])) * 100, 2)
            avg_rtt = round(np.asarray(table['latest_rtt']).mean(), 2)
            ax.text(left+0.06*(right-left), bottom+0.7*(top-bottom), 
                    f'Event Count: {count}\nTrigger Loss: {trigger} ({trigger_rate}%)\nAvg {RATE_TYPE}: {LossR}% ({LossR_}%)\nAvg INTR: {intr} (sec)\nMax CWND Diff: {cwnd_diff}%\nAvg Latency: {avg_latency}ms\nGoodput: {goodput}%\nAvg RTT: {avg_rtt}ms',
                    ha='left', fontweight='bold', fontsize=10)
                        
            if self.epoch == 'last':
                ax.set_title(f'{DIRC_TYPE} {RATE_TYPE}: {tag} | {self.model_prefix}')
            else:
                ax.set_title(f'{DIRC_TYPE} {RATE_TYPE}: {tag} | {self.model_prefix} {self.epoch} epoch')
                
            ax.set_ylabel(METS_TYPE)
            ax.set_xlabel('Relative Timestamp (sec)')
            
            # ax_twin.set_ylabel('Number of Packets Transmit')
            # ax_twin.set_yticks([])  # Remove y-axis ticks
            
            # 合併兩個圖的legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines4, labels4 = ax4.get_legend_handles_labels()
            ax.legend(lines + lines2 + lines3 + lines4, labels + labels2 + labels3 + labels4, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=5)
            fig.set_size_inches(10, 5)
            
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            
            save_path = os.path.join(self.save_path, self.model_name, 'train', 'sr', self.dirc_mets, 'models', 'plot')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            
            if self.epoch == 'last':
                save_path = os.path.join(save_path, f'{self.model_prefix}_{tag}.png')
            else:
                save_path = os.path.join(save_path, f'{self.model_prefix}_{tag}_{self.epoch}.png')
            
            print(save_path)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # plt.show()
            plt.close(fig)
    
    def save_models(self):
        save_path = os.path.join(self.save_path, self.model_name, 'train', 'sr', self.dirc_mets, 'models')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        if self.epoch == 'last':
            print('Save models:', self.model_prefix, '->', save_path)
            save_path = os.path.join(save_path, self.model_prefix)
        else:
            print('Save models:', f'{self.model_prefix}_{self.epoch}', '->', save_path)
            save_path = os.path.join(save_path, f'{self.model_prefix}_{self.epoch}')
            
        print()
        
        with open(f'{save_path}_kde_models.pkl', 'wb') as f:
            pickle.dump(self.kde_models, f)
        with open(f'{save_path}_hist_models.pkl', 'wb') as f:
            pickle.dump(self.hist_models, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{save_path}_scope_models.pkl', 'wb') as f:
            pickle.dump(self.scope_models, f)
        with open(f'{save_path}_plr_models.pkl', 'wb') as f:
            pickle.dump(self.plr_models, f)
        with open(f'{save_path}_sr_prob_models.pkl', 'wb') as f:
            pickle.dump(self.prob_models, f)