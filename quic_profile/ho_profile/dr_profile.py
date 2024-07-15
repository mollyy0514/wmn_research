import os
import pickle
import math
import copy
import numpy as np
import pandas as pd
import portion as P
import itertools as it
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from allutils import *

class DrProfile():
    def __init__(
        self, filepaths, model_prefix='Test',
        sr_model_id=None, sr_model_dscp=None,
        load_path='.', save_path='.',
        dirc_mets='dl_lost',
        sp_columns=['type'], ts_column='Timestamp',
        anchor_mode='by_event', test_mode=False,
        corr_lst=['zero', 'max', '25%', '50%', '75%']
    ):
        
        if sr_model_id is None:
            raise TypeError("請輸入單通道模型編號")
        
        self.filepaths = copy.deepcopy(filepaths)
        self.sr_model_name = sr_model_id if sr_model_dscp is None else sr_model_id + '_' + sr_model_dscp
        self.dr_model_name = 'dr_anchor_by_event'
        self.model_prefix = model_prefix
        
        self.dirc_mets = dirc_mets
        self.dirc, self.mets = dirc_mets[:2], dirc_mets[-4:]
        self.DIRC_TYPE = 'Downlink' if self.dirc == 'dl' else 'Uplink'
        self.METS_TYPE = 'Packet Loss' if self.mets == 'lost' else 'Excessive Latency'
        self.RATE_TYPE = 'PLR' if self.mets == 'lost' else 'ELR'
        self.sp_columns = sp_columns[:]
        self.ts_column = ts_column
        self.anchor_mode = anchor_mode
        self.test_mode = test_mode
        self.corr_lst = corr_lst[:]
        
        self.save_path = save_path
        print(self.save_path, self.sr_model_name, self.dr_model_name, self.model_prefix, self.dirc_mets)
        self.load_path = os.path.join(load_path, self.sr_model_name, 'train', 'sr', self.dirc_mets, 'models', self.model_prefix)
        print(self.load_path)
        
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
                self.sr_prob_models = pickle.load(f)[self.dirc_mets]
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
                self.sr_prob_models = pickle.load(f)
        
        self.corr_data = { tag: [[], []] for tag in 
            list(it.product(['LTEH', 'ENBH', 'MCGH', 'MNBH', 'SCGM', 'SCGA', 'SCGR-I', 'SCGR-II', 'SCGC-I', 'SCGC-II', 'SCGF', 'MCGF', 'NASR', 'Stable'], repeat=2)) }
        
        self.dr_prob_models = { tag: (0.5, 0.5) for tag in 
            list(it.product(['LTEH', 'ENBH', 'MCGH', 'MNBH', 'SCGM', 'SCGA', 'SCGR-I', 'SCGR-II', 'SCGC-I', 'SCGC-II', 'SCGF', 'MCGF', 'NASR', 'Stable'], repeat=2)) }
        self.dr_prob_models_table = None
        self.dr_prob_models_adjust = None
        self.dr_prob_models_adjust_table = None

        # Building Dr Model
        self.construct_profile()
        self.estimate_probability()
        self.adjust_parameters()
        self.save_models()
        for mode in corr_lst:
            _, _ = self.alter_corr_coef(mode, rho_force=None)
    
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
    
    # TODO: def anchor_by_packets
    
    def anchor_by_event(self, ho_df1, ho_df2, df2):
        scope, mets = self.scope_models, self.mets
        this_df = df2.copy()
        
        # 觀察 lost 有沒有出現在 (anchor_type, anchor_state, type) 的影響範圍內，若有的話，理論上取值的 df 不為空
        this_df = this_df[this_df[mets]].copy().reset_index(drop=True)
        
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
                    left_bound = min(max(current_left_bound, DrProfile.interpolate(prior_right_bound, current_left_bound), prior_row['end']), start_ts)
                else:
                    left_bound = min(max(current_left_bound, DrProfile.interpolate(prior_right_bound, current_left_bound), prior_row['start']), start_ts)
            else:
                left_bound = current_left_bound
            
            if post_row is not None:
                post_tag = '_'.join([s for s in post_row[self.sp_columns] if pd.notna(s)])
                post_left_bound = post_row['start'] + pd.Timedelta(seconds=(scope[post_tag][0]))
                if pd.notna(end_ts):
                    right_bound = max(min(current_right_bound, DrProfile.interpolate(current_right_bound, post_left_bound), post_row['start']), end_ts)
                else:
                    right_bound = max(min(current_right_bound, DrProfile.interpolate(current_right_bound, post_left_bound), post_row['start']), start_ts)
            else:
                right_bound = current_right_bound
            
            interval = P.closed(left_bound, right_bound)
            # if np.isinf(interval.upper) or np.isinf(interval.lower):
            #     print('unknonw inf value')
            #     continue
            try:
                ho_df1.loc[(ho_df1['start'] >= interval.lower) & (ho_df1['start'] < interval.upper), 'anchor_type'] = tag
                
                if not this_df[(this_df['Timestamp'] >= interval.lower) & (this_df['Timestamp'] < interval.upper)].empty:
                    ho_df1.loc[(ho_df1['start'] >= interval.lower) & (ho_df1['start'] < interval.upper), 'anchor_state'] = 1
                    
                # Update dataframe to accelerate the speed
                this_df = this_df[this_df[self.ts_column] >= interval.upper].copy()
            except:
                continue
            
        return ho_df1
    
    
    def setup_profile_by_event(self, df, ho_df):
        scope, mets = self.scope_models, self.mets
        this_df = df.copy()
        
        # 觀察 lost 有沒有出現在 (anchor_type, anchor_state, type) 的影響範圍內，若有的話，理論上取值的 df 不為空
        this_df = this_df[this_df[mets]].copy().reset_index(drop=True)
        
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
                    left_bound = min(max(current_left_bound, DrProfile.interpolate(prior_right_bound, current_left_bound), prior_row['end']), start_ts)
                else:
                    left_bound = min(max(current_left_bound, DrProfile.interpolate(prior_right_bound, current_left_bound), prior_row['start']), start_ts)
            else:
                left_bound = current_left_bound
            
            if post_row is not None:
                post_tag = '_'.join([s for s in post_row[self.sp_columns] if pd.notna(s)])
                post_left_bound = post_row['start'] + pd.Timedelta(seconds=(scope[post_tag][0]))
                if pd.notna(end_ts):
                    right_bound = max(min(current_right_bound, DrProfile.interpolate(current_right_bound, post_left_bound), post_row['start']), end_ts)
                else:
                    right_bound = max(min(current_right_bound, DrProfile.interpolate(current_right_bound, post_left_bound), post_row['start']), start_ts)
            else:
                right_bound = current_right_bound
            
            interval = P.closed(left_bound, right_bound)
            # if np.isinf(interval.upper) or np.isinf(interval.lower):
            #     print('unknonw inf value')
            #     continue
            
            anchor_tag = row['anchor_type']
            anchor_state = row['anchor_state']
            
            try:
                self.corr_data[(anchor_tag, tag)][0].append(anchor_state)
                if not this_df[(this_df['Timestamp'] >= interval.lower) & (this_df['Timestamp'] < interval.upper)].empty:
                    self.corr_data[(anchor_tag, tag)][1].append(1)
                else:
                    self.corr_data[(anchor_tag, tag)][1].append(0)

                # Update dataframe to accelerate the speed
                this_df = this_df[this_df[self.ts_column] >= interval.upper].copy()
            except:
                continue

    def construct_profile(self):
        scope, dirc, mets = self.scope_models, self.dirc, self.mets
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
            
            if self.anchor_mode == 'by_packet':
                df1 = self.anchor_by_packet(df1, ho_df2, df2)
                df2 = self.anchor_by_packet(df2, ho_df1, df1)
                self.setup_profile_by_packet(df1, ho_df1)
                self.setup_profile_by_packet(df2, ho_df2)
            else:
                ho_df1 = self.anchor_by_event(ho_df1, ho_df2, df2)
                ho_df2 = self.anchor_by_event(ho_df2, ho_df1, df1)
                self.setup_profile_by_event(df1, ho_df1)
                self.setup_profile_by_event(df2, ho_df2)

    def estimate_probability(self):
        for key, lst in self.corr_data.items():
            # 生成的數據
            data = {
                'first_distribution_samples': np.array(lst[0]),  # 第一個分布的樣本數據
                'second_distribution_samples': np.array(lst[1])  # 第二個分布的樣本數據
            }
            
            def log_likelihood(params, data):
                p, success_prob_when_first_is_success, success_prob_when_first_is_failure = params
                first_samples = data['first_distribution_samples']
                second_samples = data['second_distribution_samples']
                log_likelihood_val = 0.0
                for first_sample, second_sample in zip(first_samples, second_samples):
                    # 添加檢查以確保概率值在有效範圍內
                    if not 0 <= success_prob_when_first_is_success <= 1:
                        return float('inf')  # 返回無窮大值表示非法參數
                    if not 0 <= success_prob_when_first_is_failure <= 1:
                        return float('inf')  # 返回無窮大值表示非法參數
                    # 計算對數似然函數值
                    if first_sample == 1:
                        log_likelihood_val += np.log(success_prob_when_first_is_success if second_sample == 1 else (1 - success_prob_when_first_is_success))
                    else:
                        log_likelihood_val += np.log(success_prob_when_first_is_failure if second_sample == 1 else (1 - success_prob_when_first_is_failure))
                return -log_likelihood_val  # 取負對數似然函數，因為我們要最大化對數似然函數

            # 使用最大似然估計估算參數
            initial_guess = [0.5, 0.5, 0.5]  # 初始猜測值
            result = minimize(log_likelihood, initial_guess, args=(data,), method='Nelder-Mead')

            # 輸出估計的參數值
            estimated_params = result.x
            # print("Estimated parameters:")
            # print(f"p: {estimated_params[0]}")
            # print(f"success_prob_when_first_is_success: {estimated_params[1]}")
            # print(f"success_prob_when_first_is_failure: {estimated_params[2]}")
            self.dr_prob_models[key] = (estimated_params[1], estimated_params[2])
        
    
    def adjust_parameters(self):
        sr_prob_models, dr_prob_models = self.sr_prob_models, self.dr_prob_models
        
        def calculate_rho_conditional(p, q, alpha):
            """
            p := P(X = 1); P(X = 0) = 1 - p
            q := P(Y = 1); P(Y = 0) = 1 - q
            alpha := P(Y = 1 | X = 1); P(Y = 0 | X = 1) = 1 - alpha
            rho: correlation coefficient
            """
            sigma = max(math.sqrt(p * q * (1 - p) * (1 - q)), 1e-9)  # sigma_x * sigma_y
            rho = (p * alpha - p * q) / sigma
            return rho

        def rho_restriction(p, q):
            sigma = max(math.sqrt(p * q * (1 - p) * (1 - q)), 1e-9)  # sigma_x * sigma_y
            R1 = P.closed(-1, 1)  # -1 <= rho <= 1
            R2 = P.closed(-(p * q) / sigma, (1 - p * q) / sigma)  # 0 <= P(X=1, Y=1) <= 1
            R3 = P.closed((p * (1 - q) - 1) / sigma, p * (1 - q) / sigma)  # 0 <= P(X=1, Y=0) <= 1
            R4 = P.closed((q * (1 - p) - 1) / sigma, q * (1 - p) / sigma)  # 0 <= P(X=0, Y=1) <= 1
            R5 = P.closed(-((1 - p) * (1 - q)) / sigma, (1 - (1 - p) * (1 - q)) / sigma)  # 0 <= P(X=0, Y=0) <= 1
            R = R1 & R2 & R3 & R4 & R5
            return R

        def calculate_joint_probabilities(p, q, rho):
            """
            p := P(X = 1); P(X = 0) = 1 - p
            q := P(Y = 1); P(Y = 0) = 1 - q
            rho: correlation coefficient
            a := P(X = 1, Y = 1)
            b := P(X = 1, Y = 0)
            c := P(X = 0, Y = 1)
            d := P(X = 0, Y = 0)
            """
            sigma = max(math.sqrt(p * q * (1 - p) * (1 - q)), 1e-9)  # sigma_x * sigma_y
            a = p * q + rho * sigma
            b = p * (1 - q) - rho * sigma
            c = q * (1 - p) - rho * sigma
            d = (1 - p) * (1 - q) + rho * sigma
            return a, b, c, d

        def calculate_conditional_probabilities(p, q, rho):
            """
            p := P(X = 1); P(X = 0) = 1 - p
            q := P(Y = 1); P(Y = 0) = 1 - q
            rho: correlation coefficient
            alpha := P(Y = 1 | X = 1)
            beta  := P(Y = 1 | X = 0)
            gamma := P(X = 1 | Y = 1)
            delta := P(X = 1 | Y = 0)
            """
            sigma = max(math.sqrt(p * q * (1 - p) * (1 - q)), 1e-9)  # sigma_x * sigma_y
            p = min(max(1e-9, p), 1 - 1e-9)
            q = min(max(1e-9, q), 1 - 1e-9)
            alpha = (p * q + rho * sigma) / p
            beta = (q * (1 - p) - rho * sigma) / (1 - p)
            gamma = (p * q + rho * sigma) / q
            delta = (p * (1 - q) - rho * sigma) / (1 - q)
            return alpha, beta, gamma, delta

        def generate_combos(items):
            combinations_result = list(it.combinations(items, 2))
            self_combinations = [(x, x) for x in items]
            def custom_sort(item):
                return items.index(item[0]), items.index(item[1])
            combos = sorted(self_combinations + combinations_result, key=custom_sort)
            return combos

        def adjust_rho(rho, rho_limit):
            if rho > rho_limit.upper:
                return rho_limit.upper
            elif rho < rho_limit.lower:
                return rho_limit.lower
            else:
                return rho
        
        table = pd.DataFrame(columns="type1, type2, p, q, rho_lower, rho_upper, rho1, rho2, alpha, beta, gamma, delta, a1, a2, b, c, d1, d2, sum".split(", "))
        adjust_table = pd.DataFrame(columns="type1, type2, p, q, rho_lower, rho_upper, rho, alpha, beta, gamma, delta, a, b, c, d, sum".split(", "))
        items = ['LTEH', 'ENBH', 'MCGH', 'MNBH', 'SCGM', 'SCGA', 'SCGR-I', 'SCGR-II', 'SCGC-I', 'SCGC-II', 'SCGF', 'MCGF', 'NASR', 'Stable']
        combos = generate_combos(items)

        # Generate statistics table & adjust statistics table for handover event pairs
        for pair in combos:
            p = sr_prob_models[pair[0]]
            q = sr_prob_models[pair[1]]
            alpha, beta = dr_prob_models[pair]
            gamma, delta = dr_prob_models[(pair[1], pair[0])]
            a_1, a_2, b_, c_= p * alpha, q * gamma, (1-q) * delta, (1-p) * beta
            d_1, d_2 = 1 - a_1 - b_ - c_, 1 - a_2 - b_ - c_
            rho1 = calculate_rho_conditional(p, q, alpha)
            rho2 = calculate_rho_conditional(q, p, gamma)
            rho_limit = rho_restriction(p, q)
            table.loc[len(table)] = [pair[0], pair[1], p, q, rho_limit.lower, rho_limit.upper, rho1, rho2, alpha, beta, gamma, delta, a_1, a_2, b_, c_, d_1, d_2, 1.0]
            # Get the appropriate rho in [rho1, rho2]
            rho = adjust_rho(np.mean([rho1, rho2]), rho_limit)
            a, b, c, d = calculate_joint_probabilities(p, q, rho)
            alpha_adjust, beta_adjust, gamma_adjust, delta_adjust = calculate_conditional_probabilities(p, q, rho)
            adjust_table.loc[len(adjust_table)] = [pair[0], pair[1], p, q, rho_limit.lower, rho_limit.upper, rho, alpha_adjust, beta_adjust, gamma_adjust, delta_adjust, a, b, c, d, sum([a, b, c, d])]
        
        table = table.set_index(['type1', 'type2'])
        adjust_table = adjust_table.set_index(['type1', 'type2'])

        model_adjust = copy.deepcopy(dr_prob_models)
        for pair in combos:
            row = adjust_table.loc[pair]
            p, q, alpha, beta, gamma, delta = row['p'], row['q'], row['alpha'], row['beta'], row['gamma'], row['delta']
            model_adjust[pair] = (alpha, beta)
            model_adjust[(pair[1], pair[0])] = (gamma, delta)
        
        self.dr_prob_models_table = table
        self.dr_prob_models_adjust_table = adjust_table
        self.dr_prob_models_adjust = model_adjust

    def save_models(self):
        save_path = os.path.join(self.save_path, self.sr_model_name, 'train', self.dr_model_name, self.dirc_mets, 'models')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
        print('Save DR models:', self.model_prefix, '->', save_path)
        print()
        
        save_path = os.path.join(save_path, self.model_prefix)
        
        with open(f'{save_path}_dr_prob_models_mle.pkl', 'wb') as f:
            pickle.dump(self.dr_prob_models, f)
        with open(f'{save_path}_dr_prob_models_adjust.pkl', 'wb') as f:
            pickle.dump(self.dr_prob_models_adjust, f)
            
        self.dr_prob_models_table.to_csv(f'{save_path}_dr_prob_models_mle_table.csv')
        self.dr_prob_models_adjust_table.to_csv(f'{save_path}_dr_prob_models_adjust_table.csv')

    def alter_corr_coef(self, mode='zero', rho_force=None):
        sr_model_name, dr_model_name = self.sr_model_name, self.dr_model_name
        route = self.model_prefix
        dirc_mets = self.dirc_mets

        def rho_restriction(p, q):
            sigma = max(math.sqrt(p * q * (1 - p) * (1 - q)), 1e-9)  # sigma_x * sigma_y
            R1 = P.closed(-1, 1)  # -1 <= rho <= 1
            R2 = P.closed(-(p * q) / sigma, (1 - p * q) / sigma)  # 0 <= P(X=1, Y=1) <= 1
            R3 = P.closed((p * (1 - q) - 1) / sigma, p * (1 - q) / sigma)  # 0 <= P(X=1, Y=0) <= 1
            R4 = P.closed((q * (1 - p) - 1) / sigma, q * (1 - p) / sigma)  # 0 <= P(X=0, Y=1) <= 1
            R5 = P.closed(-((1 - p) * (1 - q)) / sigma, (1 - (1 - p) * (1 - q)) / sigma)  # 0 <= P(X=0, Y=0) <= 1
            R = R1 & R2 & R3 & R4 & R5
            return R

        def calculate_joint_probabilities(p, q, rho):
            """
            p := P(X = 1); P(X = 0) = 1 - p
            q := P(Y = 1); P(Y = 0) = 1 - q
            rho: correlation coefficient
            a := P(X = 1, Y = 1)
            b := P(X = 1, Y = 0)
            c := P(X = 0, Y = 1)
            d := P(X = 0, Y = 0)
            """
            sigma = max(math.sqrt(p * q * (1 - p) * (1 - q)), 1e-9)  # sigma_x * sigma_y
            a = p * q + rho * sigma
            b = p * (1 - q) - rho * sigma
            c = q * (1 - p) - rho * sigma
            d = (1 - p) * (1 - q) + rho * sigma
            return a, b, c, d

        def calculate_conditional_probabilities(p, q, rho):
            """
            p := P(X = 1); P(X = 0) = 1 - p
            q := P(Y = 1); P(Y = 0) = 1 - q
            rho: correlation coefficient
            alpha := P(Y = 1 | X = 1)
            beta  := P(Y = 1 | X = 0)
            gamma := P(X = 1 | Y = 1)
            delta := P(X = 1 | Y = 0)
            """
            sigma = max(math.sqrt(p * q * (1 - p) * (1 - q)), 1e-9)  # sigma_x * sigma_y
            p = min(max(1e-9, p), 1 - 1e-9)
            q = min(max(1e-9, q), 1 - 1e-9)
            alpha = (p * q + rho * sigma) / p
            beta = (q * (1 - p) - rho * sigma) / (1 - p)
            gamma = (p * q + rho * sigma) / q
            delta = (p * (1 - q) - rho * sigma) / (1 - q)
            return alpha, beta, gamma, delta

        def generate_combos(items):
            combinations_result = list(it.combinations(items, 2))
            self_combinations = [(x, x) for x in items]
            def custom_sort(item):
                return items.index(item[0]), items.index(item[1])
            combos = sorted(self_combinations + combinations_result, key=custom_sort)
            return combos

        def adjust_rho(rho, rho_limit):
            if rho > rho_limit.upper:
                return rho_limit.upper
            elif rho < rho_limit.lower:
                return rho_limit.lower
            else:
                return rho
        
        load_path = os.path.join('.', sr_model_name, 'train', dr_model_name, dirc_mets, 'models')
        
        try:
            with open(os.path.join(load_path, f'{route}_dr_prob_models_mle.pkl'), 'rb') as f:
                dr_prob_models = pickle.load(f)[dirc_mets]
        except:
            with open(os.path.join(load_path, f'{route}_dr_prob_models_mle.pkl'), 'rb') as f:
                dr_prob_models = pickle.load(f)
                
        dr_prob_models_table = pd.read_csv(os.path.join(load_path, f'{route}_dr_prob_models_mle_table.csv'), index_col=[0, 1])
        
        adjust_table = pd.DataFrame(columns="type1, type2, p, q, rho_lower, rho_upper, rho, alpha, beta, gamma, delta, a, b, c, d, sum".split(", "))
        items = ['LTEH', 'ENBH', 'MCGH', 'MNBH', 'SCGM', 'SCGA', 'SCGR-I', 'SCGR-II', 'SCGC-I', 'SCGC-II', 'SCGF', 'MCGF', 'NASR', 'Stable']
        combos = generate_combos(items)

        for pair in combos:
            row = dr_prob_models_table.loc[pair]
            p, q = row['p'], row['q']
            
            rho_limit = rho_restriction(p, q)
            
            if rho_force is None:
                if mode == 'zero':
                    rho_adjust = 0
                elif mode == 'max':
                    rho_adjust = rho_limit.upper
                else:
                    percentage = int(mode[:-1])
                    rho_adjust = percentage * rho_limit.upper / 100
            else:
                rho_adjust = rho_force
            
            alpha, beta, gamma, delta = calculate_conditional_probabilities(p, q, rho_adjust)
            a, b, c, d = calculate_joint_probabilities(p, q, rho_adjust)
            adjust_table.loc[len(adjust_table)] = [pair[0], pair[1], p, q, rho_limit.lower, rho_limit.upper, rho_adjust, alpha, beta, gamma, delta, a, b, c, d, sum([a, b, c, d])]
        
        adjust_table = adjust_table.set_index(['type1', 'type2'])
        
        model_adjust = copy.deepcopy(dr_prob_models)
        for pair in combos:
            row = adjust_table.loc[pair]
            p, q, alpha, beta, gamma, delta = row['p'], row['q'], row['alpha'], row['beta'], row['gamma'], row['delta']
            model_adjust[pair] = (alpha, beta)
            model_adjust[(pair[1], pair[0])] = (gamma, delta)
        
        with open(os.path.join(load_path, f'{route}_dr_prob_models_{mode}_corr.pkl'), 'wb') as f:
            pickle.dump(model_adjust, f)
        adjust_table.to_csv(os.path.join(load_path, f'{route}_dr_prob_models_{mode}_corr_table.csv'))
        
        return adjust_table, model_adjust