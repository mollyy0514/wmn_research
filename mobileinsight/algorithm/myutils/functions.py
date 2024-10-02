import os
import json
import datetime as dt
import time
import numpy as np
from threading import Timer
import datetime as dt
import traceback
import subprocess

from colorama import Fore, Style
import colorama
colorama.init() 

# Import MobileInsight modules
from mobile_insight.monitor import OnlineMonitor
from algorithm.analyzer.feature_extracter.feature_extracter import FeatureExtracter
from algorithm.analyzer.myMsgLogger import MyMsgLogger

from .predictor import Predictor

def get_ser(folder, *dev):
    d2s_path = os.path.join(folder, 'device_to_serial.json')
    with open(d2s_path, 'r') as f:
        device_to_serial = json.load(f)
        ser = []
        for d in dev:
            if d.startswith("qc"):
                ser.append(os.path.join("/dev/serial/by-id", f"usb-Quectel_RM500Q-GL_{device_to_serial[d]}-if00-port0"))
            elif d.startswith("sm"):
                ser.append(os.path.join("/dev/serial/by-id", f"usb-SAMSUNG_SAMSUNG_Android_{device_to_serial[d]}-if00-port0"))
        return tuple(ser)
# def get_ser(folder, *dev):
#     d2s_path = os.path.join(folder, 'device_to_serial.json')
#     with open(d2s_path, 'r') as f:
#         device_to_serial = json.load(f)
#         return tuple(os.path.join("/dev/serial/by-id", f"usb-SAMSUNG_SAMSUNG_Android_{device_to_serial[d]}-if00-port0") for d in dev)
    
# Show prediction result if event is predicted.
def show_predictions(dev, preds, thr = 0.5):
    
    try:
        if preds['rlf'] > thr:
            print(f'{Fore.RED}{dev} Prediction: Near RLF!!!{Style.RESET_ALL}')
    except:
        try:
            # merge eNB HO and MN HO from Table 3 and refer to them as LTE HO
            if preds['lte_cls'] > thr:
                print(f'{Fore.RED}{dev} Prediction: Near LTE_HO!!!{Style.RESET_ALL}')
        except:
            try:
                # SN HO is referred to as NR HO
                if preds['nr_cls'] > thr:
                    print(f'{Fore.RED}{dev} Prediction: Near SN_HO!!!{Style.RESET_ALL}')
            except:
                print("no preds['rlf'], preds['lte_cls'], preds['nr_cls']")

HOs = ['LTE_HO', 'MN_HO', 'SN_setup','SN_Rel', 'SN_HO', 'RLF', 'SCG_RLF']
def show_HO(dev, analyzer):

    features = analyzer.get_featuredict()
    features = {k: v for k, v in features.items() if k in HOs}
    
    for k, v in features.items():
        if v == 1:
            print(f'{dev}: HO {k} happened!!!!!')
            
selected_features = ['num_of_neis','RSRP', 'RSRQ', 'RSRP1','RSRQ1', 'nr-RSRP', 'nr-RSRQ', 'nr-RSRP1','nr-RSRQ1', 
                     'E-UTRAN-eventA3', 'eventA5', 'NR-eventA3', 'eventB1-NR-r15',
                     'LTE_HO', 'MN_HO', 'SN_setup','SN_Rel', 'SN_HO', 'RLF', 'SCG_RLF',]
def get_array_features(analyzer):
    features = analyzer.get_featuredict()
    features = {k: features[k] for k in selected_features}

    return np.array(list(features.values()))

def class_far_close(preds1, preds2, thr1=0.5, thr2=0.5):
    case1, case2 = 'Far', 'Far'
    prob1, prob2 = 0, 0
    # t_close = 3 # Threshold time for tell if the radio is close to HO
    for k in list(preds1.keys()):
        if k in ['rlf']:
            prob1 = preds1[k]
            prob2 = preds2[k]
            if preds1[k] > thr1:
                case1 = 'Close'
            if preds2[k] > thr2:
                case2 = 'Close'
    return case1, prob1, case2, prob2

# Online features collected and ML model inferring.
def device_running(dev, ser, baudrate, time_seq, time_slot, output_queue, start_sync_event, model_folder, SHOW_HO=False, record_freq=1):
    
    # Loading Model
    rlf_classifier = os.path.join(model_folder, 'rlf_cls_xgb.json')
    rlf_predictor = Predictor(rlf = rlf_classifier)
    lte_ho_classifier = os.path.join(model_folder, 'lte_HO_cls_xgb.json')
    lte_ho_predictor = Predictor(lte_cls = lte_ho_classifier)
    nr_ho_classifier = os.path.join(model_folder, 'nr_HO_cls_xgb.json')
    nr_ho_predictor = Predictor(nr_cls = nr_ho_classifier)

    src = OnlineMonitor()
    src.set_serial_port(ser)  # serial port
    src.set_baudrate(baudrate)  # baudrate of the port
    # Record time for save filename
    now = dt.datetime.today()
    t = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
    t = [x.zfill(2) for x in t]  # zero-padding to two digit
    t = '-'.join(t[:3]) + '_' + '-'.join(t[3:])
    # mi2log save path
    save_path = os.path.join('/home/wmnlab/Data/mobileinsight', f"diag_log_{dev}_{t}.mi2log")
    src.save_log_as(save_path)
    # save XML log
    dumper = MyMsgLogger()
    dumper.set_source(src)
    dumper.set_decoding(MyMsgLogger.XML) 
    dumper.set_dump_type(MyMsgLogger.FILE_ONLY)
    dumper.save_decoded_msg_as(f'/home/wmnlab/Data/XML/diag_log_{dev}_{t}.xml')
    
    feature_extracter = FeatureExtracter(mode='intensive')
    feature_extracter.set_source(src)
    # save model inference record
    save_path = os.path.join('/home/wmnlab/Data/record', f"record_{dev}_{t}.csv")
    f_out = open(save_path, 'w')
    f_out.write(','.join( ['Timestamp'] + selected_features 
                         + list(rlf_predictor.models.keys()) 
                         + list(lte_ho_predictor.models.keys()) 
                         + list(nr_ho_predictor.models.keys())) + '\n')
   
    # declare nonlocal, for convenience
    n_count = int(1/time_slot)
    n_record = int(record_freq/time_slot)
    x_ins = [ [] for _ in range(n_count)]
    
    rlf_models_num = len(rlf_predictor.models)
    lte_ho_models_num = len(lte_ho_predictor.models)
    nr_ho_models_num = len(nr_ho_predictor.models)
    
    def run_prediction(i):
        
        nonlocal x_ins
        x_in = x_ins[i]
        
        start_time = dt.datetime.now()
        # get features from mobileinsight feature_extracter
        feature_extracter.gather_intensive_L()
        feature_extracter.to_featuredict()
        features = get_array_features(feature_extracter)
        if SHOW_HO and i == (n_count-1): # show HO every with freq record_freq
            show_HO(dev, feature_extracter)
        feature_extracter.remove_intensive_L_by_time(start_time - dt.timedelta(seconds=1-time_slot))
        feature_extracter.reset()
        # if the time series length is not long enough, collect it and pass first
        if len(x_in) != (time_seq-1):
            
            x_in.append(features)
            if i == (n_count-1):
                print(f'{dev} {time_seq-len(x_in)} second after start...')

            # record
            if (i % n_record) == 0:
                w = [start_time.strftime("%Y-%m-%d %H:%M:%S.%f")] + [str(e) for e in list(features)]
                try: 
                    f_out.write(','.join(w + [''] * rlf_models_num) 
                                + ','.join(w + [''] * lte_ho_models_num) 
                                + ','.join(w + [''] * nr_ho_models_num) + '\n') 
                except: pass
            
        else:
            x_in.append(features)
            f_in = np.concatenate(x_in).flatten()
            rlf_out = rlf_predictor.foward(f_in) # inference with pre-trained model    
            lte_ho_out = lte_ho_predictor.foward(f_in)
            nr_ho_out = nr_ho_predictor.foward(f_in)
            x_in = x_in[1:]
            # record
            if (i % n_record) == 0:
                w = [start_time.strftime("%Y-%m-%d %H:%M:%S.%f")]+[str(e) for e in list(features) + list(rlf_out.values()) + list(lte_ho_out.values()) + list(nr_ho_out.values())]
                try: f_out.write(','.join(w) + ',\n')
                except: pass
            output_queue.put([dev, rlf_out, lte_ho_out, nr_ho_out, feature_extracter.cell_info])     

        x_ins[i] = x_in
        i = (i+1) % n_count
        Timer(time_slot, run_prediction, args=[i]).start()
    # Sync dual radios 
    start_sync_event.wait()
    print(f'Start {dev}.')
    # run model inference with freq time_slot
    thread = Timer(time_slot, run_prediction, args=[0])
    thread.daemon = True
    thread.start()

    try:
        src.run() # run mobileinsight to collect feature
    except:
        print(traceback.format_exc()) # if you need debug un-comment this
        f_out.close()
        time.sleep(.5)
        print(f'End {dev}.')

def send_pairs_to_phone(pairs, parent_folder, local_file_path, android_file_path, dev1, dev2):
    try:
        def convert_item(item):
            if isinstance(item, (np.float32, np.float64)):
                return float(item)
            elif isinstance(item, dict):
                return {key: (float(value) if isinstance(value, (np.float32, np.float64)) else value) for key, value in item.items()}
            return item

        converted_list = [convert_item(item) for item in pairs]

        # Step 1: Overwrite the local file with the new string
        with open(local_file_path, 'w') as f:
            json.dump(converted_list, f)

        d2s_path = os.path.join(parent_folder, 'device_to_serial.json')
        with open(d2s_path, 'r') as f:
            device_to_serial = json.load(f)
            # Step 2: Use adb to push the file to the Android device
            adb_push_dev1_cmd = f"adb -s {device_to_serial[dev1]} push {local_file_path} {android_file_path}"
            adb_push_dev2_cmd = f"adb -s {device_to_serial[dev2]} push {local_file_path} {android_file_path}"
            subprocess.run(adb_push_dev1_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(adb_push_dev2_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing adb commands: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")