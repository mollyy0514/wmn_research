#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import datetime as dt
import os
import sys
import argparse
import subprocess
from device_to_port import device_to_port
from device_to_serial import device_to_serial


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

def generate_devices_and_ports(args):
    devices = []
    serials = []
    for dev in args.devices:
        if '-' in dev:
            pmodel = dev[:2]
            start = int(dev[2:4])
            stop = int(dev[5:7]) + 1
            for i in range(start, stop):
                _dev = "{}{:02d}".format(pmodel, i)
                devices.append(_dev)
                serial = device_to_serial[_dev]
                serials.append(serial)
            continue
        devices.append(dev)
        serial = device_to_serial[dev]
        serials.append(serial)

    ports = []
    if not args.ports:
        for device in devices:
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

    return devices, serials, ports

# ===================== Main Process =====================

args = parse_arguments()
devices, serials, ports = generate_devices_and_ports(args)

length_packet = int(args.length)

if args.bitrate[-1] == 'k':
    bitrate = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bitrate = int(args.bitrate[:-1]) * 1e6
else:
    bitrate = int(args.bitrate)

total_time = args.time
number_client = args.number_client

expected_packet_per_sec = bitrate / (length_packet << 3)
sleeptime = 1.0 / expected_packet_per_sec

print(devices)
print(serials)
print(ports)
print("bitrate:", f'{args.bitrate}bps')

# ===================== Parameters =====================
HOST = '140.112.20.183'  # Lab 249
pcap_path = '/home/wmnlab/temp'

# ===================== start experiment =====================

def is_alive(p):
    if p.poll() is None:
        return True
    else:
        return False

def all_process_end(procs):
    for p in procs:
        if is_alive(p):
            return False
    return True

procs = []

for device, port, serial in zip(devices, ports, serials):
    su_cmd = 'cd sdcard/tcp_phone_exp && python3 tcp_socket_phone.py ' + \
            f'-H {HOST} -d {device} -p {port[0]} {port[1]} -b {args.bitrate} -l {args.length} -t {args.time}'
    adb_cmd = f"su -c '{su_cmd}'"
    p = subprocess.Popen([f'adb -s {serial} shell "{adb_cmd}"'], shell=True, preexec_fn = os.setpgrp)
    procs.append(p)

# ===================== wait for experiment end =====================

time.sleep(1)
while not all_process_end(procs):
    try:
        print('Alive...')
        time.sleep(1)

    except KeyboardInterrupt:
        
        su_cmd = 'pkill -2 python3'
        adb_cmd = f"su -c '{su_cmd}'"
        for serial in serials:
            subprocess.Popen([f'adb -s {serial} shell "{adb_cmd}"'], shell=True)
        
        time.sleep(5)
        # sys.exit()

print("---End Of File---")