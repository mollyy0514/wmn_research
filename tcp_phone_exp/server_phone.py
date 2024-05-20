#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import threading
import multiprocessing
import os
import sys
import datetime as dt
import argparse
import subprocess
import signal
from device_to_port import device_to_port
from socket import error as SocketError
import errno

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number_client", type=int,
                        help="number of client", default=1)
    parser.add_argument("-d", "--devices", type=str, nargs='+',
                        help="list of devices", default=["unam"])
    parser.add_argument("-p", "--ports", type=str, nargs='+',
                        help="ports to bind")
    parser.add_argument("-b", "--bitrate", type=str,
                        help="target bitrate in bits/sec (0 for unlimited)", default="1M")
    parser.add_argument("-l", "--length", type=str,
                        help="length of buffer to read or write in bytes (packet size)", default="250")
    parser.add_argument("-t", "--time", type=int,
                        help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
    return parser.parse_args()

def create_device_list(devices):
    device_list = []
    for dev in devices:
        if '-' in dev:
            pmodel = dev[:2]
            start = int(dev[2:4])
            stop = int(dev[5:7]) + 1
            for i in range(start, stop):
                _dev = "{}{:02d}".format(pmodel, i)
                device_list.append(_dev)
            continue
        device_list.append(dev)
    return device_list

def create_port_list(args, devices):
    if not args.ports:
        return [(device_to_port[device][0], device_to_port[device][1]) for device in devices]
    else:
        port_list = []
        for port in args.ports:
            if '-' in port:
                start = int(port[:port.find('-')])
                stop = int(port[port.find('-') + 1:]) + 1
                port_list.extend(list(range(start, stop)))
            else:
                port_list.append(int(port))
        return port_list

# ===================== Main Process =====================

args = parse_arguments()

devices = create_device_list(args.devices)
ports = create_port_list(args, devices)
print(devices)
print(ports)

if args.bitrate[-1] == 'k':
    bitrate = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bitrate = int(args.bitrate[:-1]) * 1e6
else:
    bitrate = int(args.bitrate)
print("bitrate:", f'{args.bitrate}bps')

length_packet = int(args.length)
expected_packet_per_sec = bitrate / (length_packet << 3)
if args.bitrate == '0':
    sleeptime = 0
else:
    sleeptime = 1.0 / expected_packet_per_sec

total_time = args.time
# ===================== Parameters =====================
HOST = '0.0.0.0'
pcap_path = '/home/wmnlab/temp'

# ===================== Global Variables =====================
stop_threads = False

os.system("echo wmnlab | sudo -S su")
# ===================== traffic capture =====================

tcpproc_list = []

def capture_traffic(devices, ports, pcap_path, current_datetime):
    for device, port in zip(devices, ports):
        pcap = os.path.join(pcap_path, f"server_pcap_BL_{device}_{port[0]}_{port[1]}_{current_datetime}_sock.pcap")
        tcpproc = subprocess.Popen([f"tcpdump -i any port '({port[0]} or {port[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)
        # tcpproc = subprocess.Popen([f"sudo tcpdump -i any port '({port[0]} or {port[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)
        tcpproc_list.append(tcpproc)
    time.sleep(1)

def kill_traffic_capture():
    print('Killing tcpdump process...')
    for tcpproc in tcpproc_list:
        os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)
        # os.system(f"sudo kill -15 {tcpproc.pid}")
    time.sleep(1)

now = dt.datetime.today()
current_datetime = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
current_datetime = [x.zfill(2) for x in current_datetime]  # zero-padding to two digit
current_datetime = '-'.join(current_datetime[:3]) + '_' + '-'.join(current_datetime[3:])

capture_traffic(devices, ports, pcap_path, current_datetime)

# ===================== setup socket =====================

def start_server(port, time):
    try:
        # Start iPerf3 server
        proc = subprocess.Popen(["iperf3", "-s", "-p", str(port)], preexec_fn=os.setpgrp)
        print(f"iPerf3 server started successfully on port {port}.")
        return proc
    except subprocess.CalledProcessError as e:
        print(f"Error starting iPerf3 server: {e}")

# Setup connections
server_proc_list = []

for dev, port in zip(devices, ports):
    try:
        # Start iPerf3 servers
        ul_server_proc = start_server(port[0])
        dl_server_proc = start_server(port[1])
        server_proc_list.append(ul_server_proc)
        server_proc_list.append(dl_server_proc)

        # Wait for keyboard interrupt
        while True:
            pass

    except KeyboardInterrupt:
        print("KeyboardInterrupt:", KeyboardInterrupt)
        stop_threads = True
# ===================== wait for experiment end =====================

def cleanup_and_exit():
    # Close sockets
    for proc in server_proc_list:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    # Kill tcpdump process
    kill_traffic_capture()

    print('Successfully closed.')

time.sleep(3)
while not stop_threads:
    try:
        time.sleep(3)
        
    except KeyboardInterrupt:
        stop_threads = True

# End without KeyboardInterrupt (Ctrl-C, Ctrl-Z)
cleanup_and_exit()
print("---End Of File---")