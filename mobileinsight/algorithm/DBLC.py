# Run algorithm with dual device by multipleprocess
# Author: Sheng-Ru Zeng
import os
import sys
import time

import multiprocessing
from multiprocessing import Process
import argparse
import random

# from myutils.at_commands import AT_Cmd_Runner
from myutils.functions import *

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
    f_cmd.write('Timestamp,R1,R2\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, nargs='+', help="device: e.g. qc00 qc01")
    parser.add_argument("-b", "--baudrate", type=int, help='baudrate', default=9600)
    args = parser.parse_args()
    baudrate = args.baudrate
    dev1, dev2 = args.device[0], args.device[1]
    ser1, ser2 = get_ser(parent_folder, *[dev1, dev2])
    
    # at_cmd_runner = AT_Cmd_Runner()
    # os.chdir(at_cmd_runner.dir_name) # cd modem utils dir to run at cmd
    
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
    #     counter = 0 # for convenience
    #     n_show = int(1/time_slot)
    #     n_record = int(record_freq/time_slot)
        while True: 
    #         # Get prediction and radio info from other multi-process
            start = time.time()
    #         outs = {}
    #         infos = {}
    #         while not output_queue.empty():
    #             pairs = output_queue.get() 
    #             outs[pairs[0]] = pairs[1]
    #             infos[pairs[0]] = pairs[2]
            
    #         if len(outs) == 2:
                
    #             out1, out2 = outs[dev1], outs[dev2]
    #             info1, info2 = infos[dev1], infos[dev2] # info format {'MN': PCI, 'earfcn': earfcn, 'band': band, 'SN': NR PCI}
    #             if (counter % n_record) == 0:
    #                 f_cmd.write(','.join([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), str(out1['rlf']), str(out2['rlf'])]) + '\n')
    #             # Show prediction result during experiment. 
    #             if counter == n_show-1:
    #                 show_predictions(dev1, out1); show_predictions(dev2, out2)
    #             counter = (counter+1) % n_show
                
    #             ################ Action Here ################
    #             # Do nothing if too close to previous action.
    #             if rest > 0:
    #                 if rest.is_integer():
    #                     print(f'Rest for {rest} more second.')
    #                 rest -= time_slot
    #             else:
    #                 case1, prob1, case2, prob2 = class_far_close(out1, out2) 
    #                 if case1 == 'Far' and case2 == 'Far':
    #                     pass # Fine, let's pass first.
                    
    #                 elif case1 == 'Far' and case2 == 'Close':
    #                     if info1['MN'] == info2['MN']: # check if dual radios share the same PCI
    #                         # random select band choice from all_band_choice1 that do not repeat with previous radio1/radio2 band
    #                         choices = [c for c in all_band_choice2 if (info1['band'] not in c and info2['band'] not in c)] 
    #                         choice =  random.sample(choices, 1)[0]  
    #                         print(f'same pci {dev1} far but {dev2} close!!!')
    #                         # record action time in f_cmd
    #                         f_cmd.write( ','.join([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), '', 'action'])+'\n' )
    #                         at_cmd_runner.change_band(dev2, choice, setting2) # action
    #                         setting2, rest = choice, rest_time # change global vars after action （necessary）
                    
    #                 elif case1 == 'Close' and case2 == 'Far':
    #                     if info1['MN'] == info2['MN']:
    #                         choices = [c for c in all_band_choice1 if (info1['band'] not in c)] 
    #                         choice =  random.sample(choices, 1)[0]
    #                         print(f'same pci {dev2} far but {dev1} close!!!')
    #                         f_cmd.write( ','.join([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), 'action', ''])+'\n' )
    #                         at_cmd_runner.change_band(dev1, choice, setting1) 
    #                         setting1, rest = choice, rest_time
                    
    #                 elif case1 == 'Close' and case2 == 'Close':         
    #                     print(f'R1/R2 both close')
    #                     if prob1 > prob2:
    #                         choices = [c for c in all_band_choice1 if (info1['band'] not in c)] 
    #                         choice =  random.sample(choices, 1)[0]
    #                         f_cmd.write( ','.join([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), 'action', ''])+'\n' )
    #                         at_cmd_runner.change_band(dev1, choice, setting1)
    #                         setting1, rest = choice, rest_time
    #                     else:
    #                         choices = [c for c in all_band_choice2 if (info1['band'] not in c and info2['band'] not in c)] 
    #                         choice =  random.sample(choices, 1)[0]
    #                         f_cmd.write( ','.join([dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), '', 'action'])+'\n' )
    #                         at_cmd_runner.change_band(dev2, choice, setting2)
    #                         setting2, rest = choice, rest_time         
    #             #############################################
                
    #         end = time.time()
    #         if time_slot - (end-start) > 0:
    #             time.sleep(time_slot-(end-start))
      
    except KeyboardInterrupt:
        # Stop Record
        print('Main process received KeyboardInterrupt')
        p1.join()
        p2.join()
        f_cmd.close()
        time.sleep(1)
        print("Process killed, closed.")
        sys.exit()