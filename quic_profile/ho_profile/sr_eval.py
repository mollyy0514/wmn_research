import os
import pickle
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import portion as P
from tqdm import tqdm
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error
from myutils import *

class Eval:
    def __init__(
        self, filepaths, model_prefix='Test',
        model_id=None, model_dscp=None, load_path='.', save_path='.', path2results=None,
        dirc_mets='dl_lost',
        sp_columns=['type'], ts_column='Timestamp',
        save_answer=False, test_mode=False,
        dataset_type='train',
    ):
        if model_id is None:
            raise TypeError("請輸入模型編號")
        
        self.iter_num = None  # number of iteration while evaluating
        
        self.filepaths = copy.deepcopy(filepaths)
        self.model_name = model_id if model_dscp is None else model_id + '_' + model_dscp
        self.model_prefix = model_prefix
        self.dataset_type = dataset_type
        
        self.dirc_mets = dirc_mets
        self.dirc, self.mets = dirc_mets[:2], dirc_mets[-4:]
        self.DIRC_TYPE = 'Downlink' if self.dirc == 'dl' else 'Uplink'
        self.METS_TYPE = 'Packet Loss' if self.mets == 'lost' else 'Excessive Latency'
        self.RATE_TYPE = 'PLR' if self.mets == 'lost' else 'ELR'
        self.sp_columns = sp_columns[:]
        self.ts_column = ts_column
        self.save_answer = save_answer
        
        self.save_path = save_path
        self.load_path = os.path.join(load_path, self.model_name, 'train', 'sr', self.dirc_mets, 'models', self.model_prefix)
        print(self.load_path)
        
        if path2results is None:
            with open(os.path.join(os.getcwd(), "quic_profile", "result_save_path.txt"), "r") as f:
                self.path2results = f.readline()
        else:
            self.path2results = path2results
        
        self.test_mode = test_mode
        
        try:
            with open(f'{self.load_path}_kde_models.pkl', 'rb') as f:
                self.kde_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.load_path}_hist_models.pkl', 'rb') as f:
                self.hist_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.load_path}_scope_models.pkl', 'rb') as f:
                self.scope_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.load_path}_plr_models.pkl', 'rb') as f:
                self.plr_models = pickle.load(f)[self.dirc_mets]
            with open(f'{self.load_path}_sr_prob_models.pkl', 'rb') as f:
                self.prob_models = pickle.load(f)[self.dirc_mets]
        except:
            with open(f'{self.load_path}_kde_models.pkl', 'rb') as f:
                self.kde_models = pickle.load(f)
            with open(f'{self.load_path}_hist_models.pkl', 'rb') as f:
                self.hist_models = pickle.load(f)
            with open(f'{self.load_path}_scope_models.pkl', 'rb') as f:
                self.scope_models = pickle.load(f)['result']
            with open(f'{self.load_path}_plr_models.pkl', 'rb') as f:
                self.plr_models = pickle.load(f)
            with open(f'{self.load_path}_sr_prob_models.pkl', 'rb') as f:
                self.prob_models = pickle.load(f)
        
        self.date, self.hms_count, self.hex_string, self.figure_id = figure_identity()
        self.save_name = f'{self.model_prefix}_{self.date}_{self.hms_count}_{self.hex_string}'
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
    
    def hist_method(self, df, ho_df):
        mets, RATE_TYPE = self.mets, self.RATE_TYPE
        scope = self.scope_models
        hist_model = self.hist_models
        prob_model = self.prob_models

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
                    left_bound = min(max(current_left_bound, Eval.interpolate(prior_right_bound, current_left_bound), prior_row['end']), start_ts)
                else:
                    left_bound = min(max(current_left_bound, Eval.interpolate(prior_right_bound, current_left_bound), prior_row['start']), start_ts)
            else:
                left_bound = current_left_bound
            
            if post_row is not None:
                post_tag = '_'.join([s for s in post_row[self.sp_columns] if pd.notna(s)])
                post_left_bound = post_row['start'] + pd.Timedelta(seconds=(scope[post_tag][0]))
                if pd.notna(end_ts):
                    right_bound = max(min(current_right_bound, Eval.interpolate(current_right_bound, post_left_bound), post_row['start']), end_ts)
                else:
                    right_bound = max(min(current_right_bound, Eval.interpolate(current_right_bound, post_left_bound), post_row['start']), start_ts)
            else:
                right_bound = current_right_bound
            
            interval = P.closed(left_bound, right_bound)

            # Concatenate PLR/ELR from mapping list
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
                
                if not Eval.generate_random_boolean(trigger_probability):
                    tmp[RATE_TYPE] = 0
            
            tmp['type'] = tag
            
            if i == 0:
                answer = tmp.copy()
            else:
                answer = pd.concat([answer, tmp], axis=0)

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
        random_bool_array = [Eval.generate_random_boolean(trigger_probability) for _ in range(len(trigger_prob_mapping))]
        trigger_prob_mapping['trigger'] = random_bool_array

        stable_df = pd.merge(stable_df, trigger_prob_mapping, on='Timestamp_sec', how='left')
        stable_df[RATE_TYPE] = stable_df[f'{RATE_TYPE}_if_trigger'] * stable_df['trigger']
        
        stable_df['type'] = 'Stable'

        del stable_df['Timestamp_sec'], stable_df[f'{RATE_TYPE}_if_trigger'], stable_df['trigger']
        
        answer = answer[answer['tx_count'].notnull()].copy()
        
        try:
            answer = pd.concat([answer, stable_df], axis=0)
            # answer = pd.concat([answer, stable_df], axis=0, ignore_index=True)
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
        answer['Y'] = answer[RATE_TYPE].apply(Eval.generate_random_boolean)
        
        eval_value = answer['Y'].mean() * 100
        ground_value = df[mets].mean() * 100
        
        answer = pd.concat([answer[['packet_number', 'Timestamp', 'type', 'relative_time', 'window_id']],
                            df[['lost', 'excl', 'loex']],
                            answer[[RATE_TYPE, 'Y']]], axis=1)
        
        return answer, eval_value, ground_value
    
    def run_hist_method(self, N=5):
        dirc, mets = self.dirc, self.mets
        RATE_TYPE = self.RATE_TYPE
        n = len(self.filepaths)
        for i, filepath in enumerate(self.filepaths):
            
            if self.test_mode and i > 1:
                break
            
            path = filepath[1] if dirc == 'dl' else filepath[2]
            print(f'{i+1}/{n}', filepath[0]); print(f'{i+1}/{n}', path)
            
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
            
            if dirc == 'dl':
                df = generate_dataframe(filepath[1], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latest_rtt'])
            else:
                df = generate_dataframe(filepath[2], sep='@', parse_dates=['Timestamp'], usecols=['packet_number', 'Timestamp', 'lost', 'excl', 'latest_rtt'])
                
            df, ho_df, empty_data = data_aligner(df, ho_df)
            
            if empty_data:
                print('*************** EMPTY DATA ***************')
                continue
            
            df['excl'] = ~df['lost'] & df['excl']
            df['loex'] = df['lost'] | df['excl']
            
            loss_rate_list = []
            answer = None
            # for iter_round in tqdm(range(N), ncols=1000):
            for iter_round in range(N):
                ans, eval_value, ground_value = self.hist_method(df, ho_df)
                
                if answer is None:
                    answer = ans.copy()
                    answer = answer.rename(columns={RATE_TYPE: f'{RATE_TYPE}_0', 'Y': f'Y_0'})
                else:
                    answer = pd.concat([answer, ans[[RATE_TYPE, 'Y']]], axis=1)
                    answer = answer.rename(columns={RATE_TYPE: f'{RATE_TYPE}_{iter_round}', 'Y': f'Y_{iter_round}'})
                
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

            # print()
            # print("Mean:", mean_value)
            # print("Standard Deviation:", std_deviation)
            # print("Ground Truth:", ground_value)
            # print("Error:", error)

            path = filepath[1] if dirc == 'dl' else filepath[2]
            
            sm_index = path.index("sm")  # 找到 "sm" 的位置
            next_slash_index = path.index("/", sm_index)  # 从 "sm" 的位置开始找到下一个斜杠 "/"
            sm_dev = path[sm_index:next_slash_index]  # 截取 "sm00" 标签
            
            sm_index = path.index("#")  # 找到 "#" 的位置
            next_slash_index = path.index("/", sm_index)  # 从 "#" 的位置开始找到下一个斜杠 "/"
            sm_trip = path[sm_index:next_slash_index]  # 截取 "#01" 标签
            
            self.records.append((loss_rate_list, mean_value, std_deviation, ground_value, error, sm_dev, sm_trip, path))
            
            # Save Answers
            if self.save_answer:
                # save_path = os.path.join(self.path2results, self.model_name, 'sr', self.dirc_mets, f'{self.save_name}_iter{N}')
                save_path = os.path.join(self.path2results, self.model_name, self.dataset_type, 'sr', self.dirc_mets, f'{self.save_name}_iter{N}')
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                    
                save_path = os.path.join(save_path, path.replace('/', '\\'))
                print(save_path)
                
                answer.to_csv(save_path, index=False)
            
            # Save Results
            # save_path = os.path.join(self.save_path, self.model_name, 'sr', self.dirc_mets, 'results')
            save_path = os.path.join(self.save_path, self.model_name, self.dataset_type, 'sr', self.dirc_mets, 'results')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
                
            save_path = os.path.join(save_path, f'{self.save_name}_iter{N}.pkl')
            print(save_path)
            
            with open(save_path, 'wb') as f:
                pickle.dump(self.records, f)
            
            # Update plot iteration number
            self.iter_num = N

    def plot(self, title=None):
        RATE_TYPE = self.RATE_TYPE
        
        if title == None:
            title = f'SR {self.DIRC_TYPE} {self.RATE_TYPE} | {self.model_prefix}'
        
        # Sample data
        x = [s[3] for s in self.records]  # Ground truths
        y = [s[1] for s in self.records]  # Mean values for evaluation
        y_error = [s[2] for s in self.records]  # Standard deviations for error bars
        
        tau, p_value = kendalltau(x, y)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(6, 4))

        # Scatter plot with error bars and horizontal caps
        ax.errorbar(x, y, yerr=y_error, linestyle='None', marker='o', color='tab:blue', capsize=5)
        ax.scatter([], [], linestyle='None', marker='o', color='tab:blue', label='Data Points')

        # Annotate RMSE From the ground truths
        rmse = np.sqrt(mean_squared_error(x, y))
        rmse_rate = rmse / np.mean(x) * 100
        slope_annotation = f'RMSE: {rmse:.3f} ({rmse_rate:.1f} %)'
        ax.annotate(slope_annotation, xy=(0.45, 0.85), xycoords='axes fraction', fontsize=13, fontstyle='italic', fontweight='bold', color='black')
        slope_annotation = f'τ (p.v.): {tau:.2f} ({p_value:.3f})'
        ax.annotate(slope_annotation, xy=(0.45, 0.80), xycoords='axes fraction', fontsize=13, fontstyle='italic', fontweight='bold', color='black')

        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        
        ax.set_xlim(0, max(x_limits[1], y_limits[1]))
        ax.set_ylim(0, max(x_limits[1], y_limits[1]))
        
        # 45-Degree Line
        ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='-', linewidth=1.1, color='tab:pink', label='45-Degree Line', alpha=0.9)

        # 创建 x 值范围
        x_limits = ax.get_xlim()  # 获取 x 的边界
        x_values = np.linspace(x_limits[0], x_limits[1], 100)  # 使用 x 的边界值作为范围

        # 计算 y = 1.1x 和 y = 0.9x 的值
        y_upper = 1.1 * x_values
        y_lower = 0.9 * x_values
        
        # 绘制 y = 1.1x 和 y = 0.9x 线
        ax.plot(x_values, y_upper, linestyle='-', linewidth=1.1, color='tab:orange', label='10%-Bound')
        ax.plot(x_values, y_lower, linestyle='-', linewidth=1.1, color='tab:orange')
        ax.fill_between(x_values, y_lower, y_upper, color='tab:orange', alpha=0.3)  # 在两条线之间填充颜色

        # Set labels and title
        ax.set_xlabel(f'{RATE_TYPE} Ground Truth')
        ax.set_ylabel(f'{RATE_TYPE} Evaluated')
        if title is not None:
            ax.set_title(title)

        # devices = [s[5] for s in self.records]
        # for i, sm_label in enumerate(devices):
        #     ax.annotate(sm_label, xy=(x[i], y[i]))
            
        # Add a legend
        ax.legend()
        fig.set_size_inches(6, 6)  # square figure
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        
        # Save figure
        # save_path = os.path.join(self.save_path, self.model_name, 'sr', self.dirc_mets, 'figures')
        save_path = os.path.join(self.save_path, self.model_name, self.dataset_type, 'sr', self.dirc_mets, 'figures')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, f'{self.save_name}_iter{self.iter_num}.png')
        
        print(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show plot
        plt.show()
        plt.close(fig)