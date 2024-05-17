# Cell phone udp socket programing

import socket
import time
import threading
import datetime as dt
import os
import sys
import argparse
import subprocess
import signal
from socket import error as SocketError
import errno


def parse_arguments():
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
    return parser.parse_args()

# ===================== Main Process =====================

args = parse_arguments()

HOST = args.host
device = args.device
ports = args.ports
print(device)
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
pcap_path = '/sdcard/experiment_log'

# ===================== Global Variables =====================
stop_threads = False

# ===================== setup socket =====================
def start_ul_client(host, port, packet_len, bitrate):
    try:
        # Start iPerf3 server
        subprocess.Popen(["iperf3", "-c", host, "-p", str(port), "-l", str(packet_len), "-b", str(bitrate)], preexec_fn = os.setpgrp)
    except subprocess.CalledProcessError as e:
        print(f"Error starting iPerf3 client for uplink: {e}")

def start_dl_client(host, port, packet_len, bitrate):
    try:
        # Start iPerf3 server
        subprocess.Popen(["iperf3", "-c", host, "-p", str(port), "-l", str(packet_len), "-b", str(bitrate), "-R"], preexec_fn = os.setpgrp)
    except subprocess.CalledProcessError as e:
        print(f"Error starting iPerf3 client for downlink: {e}")

try:
    # Setup connections
    rx_proc = start_ul_client(HOST, ports[0], length_packet, bitrate)
    tx_proc = start_dl_client(HOST, ports[1], length_packet, bitrate)
except KeyboardInterrupt:
    print("KeyboardInterrupt:", KeyboardInterrupt)
    stop_threads = True

print(f'Create UL socket for {device} at {HOST}:{ports[0]}.')
print(f'Create DL socket for {device} at {HOST}:{ports[1]}.')

time.sleep(1)

# ===================== traffic capture =====================

if not os.path.isdir(pcap_path):
   os.system(f'mkdir {pcap_path}')

now = dt.datetime.today()
current_datetime = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
current_datetime = [x.zfill(2) for x in current_datetime]  # zero-padding to two digit
current_datetime = '-'.join(current_datetime[:3]) + '_' + '-'.join(current_datetime[3:])

pcap = os.path.join(pcap_path, f"client_pcap_BL_{device}_{ports[0]}_{ports[1]}_{current_datetime}_sock.pcap")
tcpproc = subprocess.Popen([f"tcpdump -i any port '({ports[0]} or {ports[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)
time.sleep(1)

# ===================== wait for experiment end =====================

def cleanup_and_exit():
    # Close sockets
    os.killpg(os.getpgid(rx_proc.pid), signal.SIGTERM)
    os.killpg(os.getpgid(tx_proc.pid), signal.SIGTERM)
    # Kill tcpdump process
    os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)
    print(f'{device} successfully closed.')

time.sleep(3)
while not stop_threads:
    try:
        time.sleep(3)
        
    except KeyboardInterrupt:
        stop_threads = True

# End without KeyboardInterrupt (Ctrl-C, Ctrl-Z)
cleanup_and_exit()
print(f"---End Of File ({device})---")