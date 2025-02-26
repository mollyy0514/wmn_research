# Cell phone udp socket programing

import socket
import json
import time
import threading
import datetime as dt
import os
import sys
import argparse
import subprocess
import signal

#=================argument parsing======================
parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host", type=str,
                    help="server ip address", default="140.112.20.183")   # Lab249 外網
parser.add_argument("-d", "--device", type=str,   
                    help="device", default=["unam"])
parser.add_argument("-p", "--ports", type=int, nargs='+',    # input list of port numbers sep by 'space'
                    help="list of ul/dl port to bind")
parser.add_argument("-b", "--bitrate", type=str,
                    help="target bitrate in bits/sec (0 for unlimited)", default="1M")
parser.add_argument("-l", "--length", type=str,
                    help="length of buffer to read or write in bytes (packet size)", default="250")
parser.add_argument("-t", "--time", type=int,
                    help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
parser.add_argument("-e", "--emulator", type=bool,
                    help="whether the system is running emulation or not", default=False)
args = parser.parse_args()


#=================other variables========================
HOST = args.host  # Lab 249
dev = args.device
ports = args.ports
print(HOST, ports)

if args.bitrate[-1] == 'k':
    bitrate = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bitrate = int(args.bitrate[:-1]) * 1e6
else:
    bitrate = int(args.bitrate)
print("bitrate:", args.bitrate)

length_packet = int(args.length)
expected_packet_per_sec = bitrate / (length_packet << 3)
if args.bitrate == '0':
    sleeptime = 0
else:
    sleeptime = 1.0 / expected_packet_per_sec

total_time = args.time
#=================gloabal variables======================
stop_threads = False

# Function define
def give_server_DL_addr():
    
    outdata = 'hello'
    rx_socket.sendto(outdata.encode(), (HOST, ports[1]))

def receive(s, dev): # s should be rx_socket

    global stop_threads

    seq = 1
    prev_receive = 1
    time_slot = 1

    while not stop_threads:
        try:

            indata, _ = s.recvfrom(1024)

            try: start_time
            except NameError:
                start_time = time.time()

            if len(indata) != length_packet:
                print("packet with strange length: ", len(indata))

            seq = int(indata.hex()[32:40], 16)
            ts = int(int(indata.hex()[16:24], 16)) + float("0." + str(int(indata.hex()[24:32], 16)))

            # # Show information
            # if time.time()-start_time > time_slot:
            #     print(f"{dev} [{time_slot-1}-{time_slot}]", "receive", seq-prev_receive)
            #     time_slot += 1
            #     prev_receive = seq
            
        except KeyboardInterrupt:
            print('Manually interrupted.')
            stop_threads = True

        except Exception as inst:
            print("Error: ", inst)
            stop_threads = True

def transmit(s):

    global stop_threads
    print("start transmission: ")
    
    seq = 1
    prev_transmit = 0
    
    start_time = time.time()
    next_transmit_time = start_time + sleeptime
    
    time_slot = 1
    
    while time.time() - start_time < total_time and not stop_threads:
        try:
            t = time.time()
            data_list = []
            while t < next_transmit_time:
                data_list = read_info_file(data_list)
                t = time.time()
            next_transmit_time = next_transmit_time + sleeptime

            euler = 271828
            pi = 31415926
            datetimedec = int(t)
            microsec = int((t - int(t))*1000000)

            # random data
            redundant = os.urandom(length_packet-4*5)
            # outdata = euler.to_bytes(4, 'big') + pi.to_bytes(4, 'big') + datetimedec.to_bytes(4, 'big') + microsec.to_bytes(4, 'big') + seq.to_bytes(4, 'big') + redundant

            # change random data to mobileinsight info pairs
            if data_list != []:
                data_str = json.dumps(data_list)
                data_bytes = data_str.encode('utf-8')
                outdata = euler.to_bytes(4, 'big') + pi.to_bytes(4, 'big') + datetimedec.to_bytes(4, 'big') + microsec.to_bytes(4, 'big') + seq.to_bytes(4, 'big')
                outdata += data_bytes
                # random data
                redundant = os.urandom(length_packet - 4*5 - len(data_str))
                outdata += redundant
            else:
                # random data
                redundant = os.urandom(length_packet-4*5)
                outdata = euler.to_bytes(4, 'big') + pi.to_bytes(4, 'big') + datetimedec.to_bytes(4, 'big') + microsec.to_bytes(4, 'big') + seq.to_bytes(4, 'big') + redundant

            # send the outdata
            s.sendto(outdata, (HOST, ports[0]))
            seq += 1
        
            # if time.time()-start_time > time_slot:
            #     print("[%d-%d]"%(time_slot-1, time_slot), "transmit", seq-prev_transmit)
            #     time_slot += 1
            #     prev_transmit = seq

            if len(data_list) == 6:
                try:
                    ct = dt.datetime.today()
                    tmp_record_file = ""
                    if args.emulator == False:
                        tmp_record_file = os.path.join("/sdcard/Data", f"{ct.year}{ct.month:02d}{ct.day:02d}_{dev}_tmp_record.txt")
                    else:
                        tmp_record_file = os.path.join("/home/wmnlab/temp", f"{ct.year}{ct.month:02d}{ct.day:02d}_{dev}_tmp_record.txt")
                    # Write it in for QUIC to read
                    with open(tmp_record_file, 'w') as file:
                        file.write(f'{ct.strftime("%Y-%m-%d %H:%M:%S.%f")}@')
                        # Iterate through the list and write each item on a new line
                        for i in range(len(data_list)):
                            if i < len(data_list) - 1:
                                file.write(f"{data_list[i]}@")
                            else:
                                file.write(f"{data_list[i]}")
                except:
                    data_str = ""

        except Exception as e:
            print(ports[0], e)
            print(ports[0], data_str)
            stop_threads = True
    stop_threads = True
    print("---transmission timeout---")
    print("transmit", seq, "packets")

def read_info_file(data_list):
    if args.emulator == False:
        record_file_path = "/sdcard/Data/record_pair.json"
    else:
        record_file_path = "/home/wmnlab/Data/record_pair.json"
    # only one row once
    try:
        with open(record_file_path, newline='') as f:
            data = json.load(f)
            data_list = data
    except:
        data_list = []
    return data_list

# Create DL receive and UL transmit multi-client sockets
rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f'Create DL socket.')
print(f'Create UL socket.')

# Transmit data from receive socket to server DL port to let server know addr first
while True:
    give_server_DL_addr()
    break

# Start subprocess of tcpdump
now = dt.datetime.today()
n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
n = [x.zfill(2) for x in n]  # zero-padding to two digit
n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
if args.emulator == False:
    pcap_path = f"/sdcard/experiment_log/{n[:10]}/client_pcap"
else:
    pcap_path = f"/home/wmnlab/Desktop/experiment_log/{n[:10]}/client_pcap"
if not os.path.isdir(pcap_path):
   print("makedir: {pcap_path}")
   os.makedirs(pcap_path)

pcap = os.path.join(pcap_path, f"client_pcap_BL_{dev}_{ports[0]}_{ports[1]}_{n}_sock.pcap")
tcpproc = subprocess.Popen([f"tcpdump -i any port '({ports[0]} or {ports[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)

time.sleep(1)

# Create and start DL receive multipleprocess
t_rx = threading.Thread(target=receive, args=(rx_socket, dev, ), daemon=True)
t_rx.start()

# Create and start UL transmission multiprocess
t_tx = threading.Thread(target=transmit, args=(tx_socket,), daemon=True)
t_tx.start()

def cleanup_and_exit():
    # Close sockets
    tx_socket.close()
    rx_socket.close()

    # Kill tcpdump process
    os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)

    print(f'{dev} successfully closed.')

time.sleep(3)
while not stop_threads:
    try:
        time.sleep(3)
        
    except KeyboardInterrupt:
        stop_threads = True

# End without KeyboardInterrupt (Ctrl-C, Ctrl-Z)
cleanup_and_exit()
print(f"---End Of File ({dev})---")