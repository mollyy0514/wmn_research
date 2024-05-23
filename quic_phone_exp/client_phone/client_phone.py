import time
import datetime as dt
import os
import sys
import argparse
import subprocess
from device_to_port import device_to_port
from device_to_serial import device_to_serial
from adbutils import adb

#=================argument parsing======================
parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host", type=str,
                    help="server ip address", default="140.112.20.183")   # Lab249 外網
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
print(serials)
print(ports)

devices_adb = []
for device, serial in zip(devices, serials):
    devices_adb.append(adb.device(serial))
print(devices_adb)

#=================other variables====================
HOST = args.host # Lab 249
bitrate = args.bitrate
length = args.length
total_time = args.time

#=================start sockets=======================
procs = []
for device_adb, device, port, serial in zip(devices_adb, devices, ports, serials):
    print(device, serial, "\n")
    portString = f"{port[0]},{port[1]}"
    device_adb.shell("su -c 'cd /data/data/com.termux/files/home/wmn_research && chmod +x ./quic_phone_exp/socket/quic_tcpdump.py'")
    client_tcpdump_cmd = f"cd /data/data/com.termux/files/home/wmn_research && python3 ./quic_phone_exp/socket/quic_tcpdump.py -d {device} -p {portString}"
    adb_tcpdump_cmd = f"su -c '{client_tcpdump_cmd}'"
    
    device_adb.shell("su -c 'cd /data/data/com.termux/files/home/wmn_research && chmod +x ./quic_phone_exp/client_phone/client_socket.sh'")
    su_cmd = f'cd /data/data/com.termux/files/home/wmn_research && ./quic_phone_exp/client_phone/client_socket.sh {device} {portString} {total_time} {bitrate} {length}'
    adb_cmd = f"su -c '{su_cmd}'"

    p_tcpdump = subprocess.Popen([f'adb -s {serial} shell "{adb_tcpdump_cmd}"'], shell=True, preexec_fn=os.setpgrp)
    p_socket = subprocess.Popen([f'adb -s {serial} shell "{adb_cmd}"'], shell=True, preexec_fn = os.setpgrp)

    procs.append(p_tcpdump)

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

while not all_process_end(procs):
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        su_cmd = 'pkill -2 python3'
        adb_cmd = f"su -c '{su_cmd}'"
        for serial in serials:
            subprocess.Popen([f'adb -s {serial} shell "{adb_cmd}"'], shell=True)

print("---End Of File---")
