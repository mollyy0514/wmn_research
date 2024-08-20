# run if the mobile phone has been turned off
# or if we want to git pull the latest version
from adbutils import adb
import os
import sys
import subprocess
import argparse

device_to_serial = {
    "sm00": "R5CR20FDXHK",
    "sm01": "R5CR30P9Z8Y",
    "sm02": "R5CRA1GCHFV",
    "sm03": "R5CRA1JYYQJ",
    "sm04": "R5CRA1EV0XH",
    "sm05": "R5CRA1GBLAZ",
    "sm06": "R5CRA1ESYWM",
    "sm07": "R5CRA1ET22M",
    "sm08": "R5CRA1D23QK",
    "sm09": "R5CRA2EGJ5X",
}

device_to_port = {
    "sm00": [5200, 5201],
    "sm01": [5202, 5203],
    "sm02": [5204, 5205],
    "sm03": [5206, 5207],
    "sm04": [5208, 5209],
    "sm05": [5210, 5211],
    "sm06": [5212, 5213],
    "sm07": [5214, 5215],
    "sm08": [5216, 5217],
    "sm09": [5218, 5219],
}

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--devices", type=str, nargs='+', required=True,
                    help="device names followed by the branch name (e.g., 'sm00 main sm01 bbr_sender')")
args = parser.parse_args()

device_branch_pairs = list(zip(args.devices[::2], args.devices[1::2]))

for device_name, branch in device_branch_pairs:
    if adb.list()[0].state != "unauthorized":
        if device_name:
            device = adb.device(device_to_serial[device_name])
        else:
            device = adb.device()
    else:
        print(device_name, "adbutils.errors.AdbError: device unauthorized.")
        sys.exit(1)

    print(device_name, device)
    print("-----------------------------------")

    tools = ["git", "iperf3m", "iperf3", "python3", "tcpdump", "tmux", "vim"]  
    print(device.shell("su -c 'cd /sdcard/wmnl-handoff-research && /data/git pull'"))
    if device_name[:2] == "sm": 
        device.shell("su -c 'mount -o remount,rw /system/bin'")
        for tool in tools:
            device.shell("su -c 'cp /data/{} /bin'".format(tool))
            device.shell("su -c 'chmod +x /bin/{}'".format(tool))
    print("-----------------------------------")

    # git pull the latest version and go build
    print(device_name, device.shell("su -c 'cd /data/data/com.termux/files/home/wmn_research && /data/git restore . && /data/git pull && /data/git checkout {}'".format(branch)))

    # UDP_phone_exp
    su_cmd = 'rm -rf /sdcard/udp_phone_exp && cp -r /data/data/com.termux/files/home/wmn_research/udp_phone_exp /sdcard'
    adb_cmd = f"su -c '{su_cmd}'"
    device.shell(adb_cmd)
    print(device_name, 'Update udp_phone_exp!')
    print("-----------------------------------")

    # TCP_phone_exp
    su_cmd = 'rm -rf /sdcard/tcp_phone_exp && cp -r /data/data/com.termux/files/home/wmn_research/tcp_phone_exp /sdcard'
    adb_cmd = f"su -c '{su_cmd}'"
    device.shell(adb_cmd)
    print(device_name, 'Update tcp_phone_exp!')
    print("-----------------------------------")

print('---End Of File---')
