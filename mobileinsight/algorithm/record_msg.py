# Run algorithm with dual device by multipleprocess
# Author: Sheng-Ru Zeng
import os
import sys
import time

import multiprocessing
from multiprocessing import Process
import argparse
import random

from algorithm.myutils.at_commands import AT_Cmd_Runner
from algorithm.myutils.functions import *

if __name__ == "__main__":
    # get file path
    script_folder = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(script_folder)
    # f_cmd is for record action time
    now = dt.datetime.today()
    t = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
    t = [x.zfill(2) for x in t]  # zero-padding to two digit
    t = '-'.join(t[:3]) + '_' + '-'.join(t[3:])
    f = os.path.join('/home/wmnlab/Data/command_Time', f'{t}_cmd_record.csv')
    f_cmd = open(f,mode='w')
    f_cmd.write('Timestamp,RLF_R1,RLF_R2,LTE_HO_R1,LTE_HO_R2,NR_HO_R1,NR_HO_R2\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, nargs='+', help="device: e.g. qc00 qc01")
    parser.add_argument("-b", "--baudrate", type=int, help='baudrate', default=9600)
    args = parser.parse_args()
    baudrate = args.baudrate
    dev1, dev2 = args.device[0], args.device[1]
    ser1, ser2 = get_ser(parent_folder, *[dev1, dev2])
    # dev1 = args.device[0]
    # ser = get_ser(parent_folder, *[dev1])
    # ser1 = ser[0]

    # global variable
    # setting1, setting2 = at_cmd_runner.query_band(dev1), at_cmd_runner.query_band(dev2)
    time_seq = 8 # Read Coefficients
    time_slot = 0.01 # Decide action frequency (second)
    record_freq = 0.1 # record file frequency (sec)
    rest_time = 5.0 # rest how many second.
    rest = 0 # for convenience
    
    all_band_choice1 = [ '3', '7', '1:3:7:8', '1:3', '3:7', '3:8', '7:8', '1:7',
                        '1:3:7', '1:3:8', '1:7:8', '3:7:8']
    all_band_choice2 = [ '3', '7', '8', '1:3:7:8', '1:3', '3:7', '3:8', '7:8', '1:7', '1:8',
                        '1:3:7', '1:3:8', '1:7:8', '3:7:8']
    
    # multipleprocessing
    output_queue = multiprocessing.Queue()
    start_sync_event = multiprocessing.Event()
    time.sleep(.2)
    
    SHOW_HO = True # whether to show HO in terminal window
    model_folder = os.path.join(parent_folder, 'model') # model path
    # using multi-processing to run prediction model inference on both dual radios
    p1 = Process(target=device_running, args=[dev1, ser1, baudrate, time_seq, time_slot, output_queue, start_sync_event, model_folder, SHOW_HO, record_freq])     
    p2 = Process(target=device_running, args=[dev2, ser2, baudrate, time_seq, time_slot, output_queue, start_sync_event, model_folder, SHOW_HO, record_freq])
    p1.start()
    p2.start()
    
    # Sync two device.
    time.sleep(3)
    start_sync_event.set()
    time.sleep(.1)
    
    # Main Process
    try:
        counter = 0 # for convenience
        n_show = int(1/time_slot)
        n_record = int(record_freq/time_slot)
        while True: 
            # Get prediction and radio info from other multi-process
            start = time.time()
            outs = {}
            infos = {}
            while not output_queue.empty():
                pairs = output_queue.get() 
                
                local_file = os.path.join('/home/wmnlab/Data', f'record_pair.json')
                android_file = os.path.join('/sdcard/Data', f'record_pair.json')
                # Sending the info pairs to the Android device
                send_pairs_to_phone(pairs, parent_folder, local_file, android_file, dev1, dev2)
                
                outs[pairs[0]] = [pairs[1], pairs[2], pairs[3]]  # the probability of RLF
                infos[pairs[0]] = pairs[4]  # info format {'MN': PCI, 'earfcn': earfcn, 'band': band, 'SN': NR PCI}
            
            if len(outs) == 2:
                rlf_out1, rlf_out2 = outs[dev1][0], outs[dev2][0]
                lte_ho_out1, lte_ho_out2 = outs[dev1][1], outs[dev2][1]
                nr_ho_out1, nr_ho_out2 = outs[dev1][2], outs[dev2][2]
                info1, info2 = infos[dev1], infos[dev2]
                if (counter % n_record) == 0:
                    f_cmd.write(','.join([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), 
                                          str(rlf_out1['rlf']), str(rlf_out2['rlf']), 
                                          str(lte_ho_out1['lte_cls']), str(lte_ho_out2['lte_cls']), 
                                          str(nr_ho_out1['nr_cls']), str(nr_ho_out2['nr_cls'])]) + '\n')
                # Show prediction result during experiment. 
                if counter == n_show-1:
                    show_predictions(dev1, rlf_out1); show_predictions(dev2, rlf_out2)
                    show_predictions(dev1, lte_ho_out1); show_predictions(dev2, lte_ho_out2)
                    show_predictions(dev1, nr_ho_out1); show_predictions(dev2, nr_ho_out2)
                counter = (counter+1) % n_show

            end = time.time()
            if time_slot - (end-start) > 0:
                time.sleep(time_slot-(end-start))
      
    except KeyboardInterrupt:
        # Stop Record
        print('Main process received KeyboardInterrupt')
        p1.join()
        p2.join()
        f_cmd.close()
        time.sleep(1)
        print("Process killed, closed.")
        sys.exit()