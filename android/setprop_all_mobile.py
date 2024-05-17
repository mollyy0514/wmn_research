#!/usr/bin/env python3

# run if the mobile device is reconnected

# Command Usage:
# pip3 install adbutils
# ./auto_mi_iperf.py

from adbutils import adb
import os
import sys

serial_to_device = {
    "R5CR20FDXHK":"sm00",
    "R5CR30P9Z8Y":"sm01",
    "R5CRA1GCHFV":"sm02",
    "R5CRA1JYYQJ":"sm03",
    "R5CRA1EV0XH":"sm04",
    "R5CRA1GBLAZ":"sm05",
    "R5CRA1ESYWM":"sm06",
    "R5CRA1ET22M":"sm07",
    "R5CRA1D23QK":"sm08",
    "R5CRA2EGJ5X":"sm09",
    "76857c8" :"qc00",
    "bc4587d" :"qc01",
    "5881b62f":"qc02",
    "32b2bdb2":"qc03",
}

os.system(f"echo {sys.argv[1]} | sudo -S su")

devices_info = []
for i, info in enumerate(adb.list()):
    try:
        if info.state == "device":
            # <serial> <device|offline> <device name>
            devices_info.append((info.serial, info.state, serial_to_device[info.serial]))
        else:
            print("Unauthorized device {}: {} {}".format(serial_to_device[info.serial], info.serial, info.state))
    except:
        print("Unknown device: {} {}".format(info.serial, info.state))

devices_info = sorted(devices_info, key=lambda v:v[2])

devices = []
for i, info in enumerate(devices_info):
    devices.append(adb.device(info[0]))
    print("{} - {} {} {}".format(i+1, info[0], info[1], info[2]))
print("-----------------------------------")

for info in adb.list():
    if info.state == "unauthorized":
        sys.exit(1)

# setprop
for device, info in zip(devices, devices_info):
    print(info[2], device.shell("su -c 'getprop sys.usb.config'"))
    print(info[2], device.shell("su -c 'setprop sys.usb.config diag,serial_cdev,rmnet,adb'"))
