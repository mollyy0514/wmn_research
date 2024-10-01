#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import threading
import multiprocessing
import os
import sys
import json
import datetime as dt
import argparse
import subprocess
import signal
from device_to_port import device_to_port


#=================argument parsing======================
parser = argparse.ArgumentParser()  
parser.add_argument("-d", "--devices", type=str, nargs='+',   # input list of devices sep by 'space'
                    help="list of devices", default=["unam"])
parser.add_argument("-p", "--ports", type=str, nargs='+',     # input list of port numbers sep by 'space'
                    help="ports to bind")
parser.add_argument("-b", "--bitrate", type=str,
                    help="target bitrate in bits/sec (0 for unlimited)", default="1M")
parser.add_argument("-l", "--length", type=str,
                    help="length of buffer to read or write in bytes (packet size)", default="250")
parser.add_argument("-t", "--time", type=int,
                    help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
args = parser.parse_args()

# Get argument devices
devices = []
for dev in args.devices:
    if '-' in dev:
        pmodel = dev[:2]
        start = int(dev[2:4])
        stop = int(dev[5:7]) + 1
        for i in range(start, stop):
            _dev = "{}{:02d}".format(pmodel, i)
            devices.append(_dev)
        continue
    devices.append(dev)

# Get the corresponding ports with devices
ports = []
if not args.ports:
    for device in devices:
        # default uplink port and downlink port for each device
        ports.append((device_to_port[device][0], device_to_port[device][1]))  
else:
    for port in args.ports:
        if '-' in port:
            start = int(port[:port.find('-')])
            stop = int(port[port.find('-') + 1:]) + 1
            for i in range(start, stop):
                ports.append(i)
            continue
        ports.append(int(port))

print(devices)
print(ports)

if args.bitrate[-1] == 'k':
    bitrate = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bitrate = int(args.bitrate[:-1]) * 1e6
else:
    bitrate = int(args.bitrate)
print("bitrate:", bitrate)

length_packet = int(args.length)
expected_packet_per_sec = bitrate / (length_packet << 3)
if args.bitrate == '0':
    sleeptime = 0
else:
    sleeptime = 1.0 / expected_packet_per_sec

total_time = args.time
#=================other variables========================
HOST = '0.0.0.0' # 140.112.20.183
#=================global variables=======================
stop_threads = False
udp_addr = {}

# Function define

def fill_udp_addr(s):
    indata, addr = s.recvfrom(1024)
    udp_addr[s] = addr 

def receive(s, dev, port, f_cmd):
    global stop_threads
    print(f"wait for indata from {dev} at {port}...")

    seq = 1
    prev_receive = 1
    time_slot = 1

    while not stop_threads:
        try:
            # receive data, update client's addresses (after receiving, server know where to transmit)
            indata, addr = s.recvfrom(1024)

            try: start_time
            except NameError:
                start_time = time.time()

            if len(indata) != length_packet:
                print("packet with strange length: ", len(indata))

            seq = int(indata.hex()[32:40], 16)
            ts = int(int(indata.hex()[16:24], 16)) + float("0." + str(int(indata.hex()[24:32], 16)))

            # decode the info record pair data
            fixed_size = 4 * 5
            data_bytes = indata[fixed_size:]
            for i in range(100, len(data_bytes)):
                try:
                    curr = bytes([data_bytes[i]]).decode('utf-8')
                    next = bytes([data_bytes[i+1]]).decode('utf-8')
                except Exception as e:
                    data_bytes = data_bytes[:i]
                    break
            now = dt.datetime.now()
            tmp_record_file = os.path.join("/home/wmnlab/temp", f"{now.year}{now.month:02d}{now.day:02d}_{dev}_tmp_record.txt")
            try:
                data_str = data_bytes.decode('utf-8')
                data_list = json.loads(data_str)
                if data_list[0] == dev:
                    # Write in the records
                    f_cmd.write(','.join([now.strftime("%Y-%m-%d %H:%M:%S.%f"), str(data_list[1]['rlf']), str(data_list[2]['lte_cls']), str(data_list[3]['nr_cls']),
                                        str(data_list[4]['MN']), str(data_list[4]['earfcn']), str(data_list[4]['band']), str(data_list[4]['SN'])]) + '\n')
                    # Write it in for QUIC to read
                    with open(tmp_record_file, 'w') as file:
                        file.write(f'{now.strftime("%Y-%m-%d %H:%M:%S.%f")},')
                        # Iterate through the list and write each item on a new line
                        for i in range(len(data_list)):
                            if i < len(data_list) - 1:
                                file.write(f"{data_list[i]},")
                            else:
                                file.write(f"{data_list[i]}")
            except:
                data_str = ""

            # Show information
            if time.time()-start_time > time_slot:
                print(f"{dev}:{port} [{time_slot-1}-{time_slot}]", "receive", seq-prev_receive)
                time_slot += 1
                prev_receive = seq

        except Exception as inst:
            print("Error: ", inst)
            stop_threads = True

def transmit(sockets):

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
            while t < next_transmit_time:
                t = time.time()
            next_transmit_time = next_transmit_time + sleeptime

            euler = 271828
            pi = 31415926
            datetimedec = int(t)
            microsec = int((t - int(t))*1000000)

            redundant = os.urandom(length_packet-4*5)
            outdata = euler.to_bytes(4, 'big') + pi.to_bytes(4, 'big') + datetimedec.to_bytes(4, 'big') + microsec.to_bytes(4, 'big') + seq.to_bytes(4, 'big') + redundant
            
            for s in sockets:
                if s in udp_addr.keys():
                    s.sendto(outdata, udp_addr[s])
            seq += 1
        
            if time.time()-start_time > time_slot:
                print("[%d-%d]"%(time_slot-1, time_slot), "transmit", seq-prev_transmit)
                time_slot += 1
                prev_transmit = seq

        except Exception as e:
            print(e)
            stop_threads = True
    stop_threads = True
    print("---transmission timeout---")
    print("transmit", seq, "packets")

# Set up UL receive /  DL transmit sockets for multiple clients
rx_sockets = []
tx_sockets = []
for dev, port in zip(devices, ports):
    s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s1.bind((HOST, port[0]))
    rx_sockets.append(s1)
    s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s2.bind((HOST, port[1]))
    tx_sockets.append(s2)
    print(f'Create socket at {HOST}:{port[0]} for UL...')
    print(f'Create socket at {HOST}:{port[1]} for DL...')

# Get client addr with server DL port first
t_fills = []
for s in tx_sockets:
    t = threading.Thread(target=fill_udp_addr, args=(s, ))
    t.start()
    t_fills.append(t)

print('Wait for filling up client address first...')
for t in t_fills:
    t.join()
print('Successful get udp addr!')

# experiment data record files
now = dt.datetime.today()
n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
n = [x.zfill(2) for x in n]  # zero-padding to two digit
n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
pcap_path = '/home/wmnlab/temp'
# record info pairs file
f1 = os.path.join(pcap_path, f'{now.year}{now.month:02d}{now.day:02d}_{devices[0]}_cmd_record.csv')
f1_cmd = open(f1,mode='w')
f1_cmd.write('Timestamp,rlf,lte_cls,nr_cls,MN,earfcn,band,SN\n')
f2 = os.path.join(pcap_path, f'{now.year}{now.month:02d}{now.day:02d}_{devices[1]}_cmd_record.csv')
f2_cmd = open(f2,mode='w')
f2_cmd.write('Timestamp,rlf,lte_cls,nr_cls,MN,earfcn,band,SN\n')
# Start subprocess of tcpdump
tcpproc_list = []
for device, port in zip(devices, ports):
    pcap = os.path.join(pcap_path, f"server_pcap_BL_{device}_{port[0]}_{port[1]}_{n}_sock.pcap")
    tcpproc =  subprocess.Popen([f"sudo tcpdump -i any port '({port[0]} or {port[1]})' -w {pcap}"], shell=True, preexec_fn = os.setpgrp)
    tcpproc_list.append(tcpproc)    
time.sleep(1)

# Create and start UL receive multi-thread
rx_threads = []
for s, dev, port in zip(rx_sockets, devices, ports):
    if (dev == devices[0]):
        t_rx = threading.Thread(target = receive, args=(s, dev, port[0], f1_cmd), daemon=True)
    else:
        t_rx = threading.Thread(target = receive, args=(s, dev, port[0], f2_cmd), daemon=True)
    rx_threads.append(t_rx)
    t_rx.start()

# Start DL transmission multipleprocessing
p_tx = multiprocessing.Process(target=transmit, args=(tx_sockets,), daemon=True)
p_tx.start()

time.sleep(3)
while not stop_threads:
    try:
        time.sleep(3)
        
    except KeyboardInterrupt:
        stop_threads = True
        time.sleep(1)

# Kill transmit process
p_tx.terminate()
time.sleep(1)

# Close sockets
for s1, s2 in zip(tx_sockets, rx_sockets):
    s1.close()
    s2.close()

# Kill tcpdump process
print('Killing tcpdump process...')
for tcpproc in tcpproc_list:
    os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)
time.sleep(1)

print('Successfully closed.')
print("---End Of File---")