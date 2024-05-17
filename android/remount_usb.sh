#! /usr/bin/env bash

# Edit File: "/etc/modprobe.d/blacklist.conf"
"""
### sudo vim /etc/modprobe.d/blacklist.conf
blacklist pl2303
### sudo systemctl restart systemd-modules-load.service
### sudo reboot (optional)
### However, gps module needs pl2303!
""" 

# https://superuser.com/questions/572034/how-to-restart-ttyusb
# https://www.cyberciti.biz/faq/restarting-ntp-service-on-linux/
# https://superuser.com/questions/154587/is-there-a-way-to-refresh-the-current-configuration-used-by-modprobe-with-a-newl
# https://www.cnblogs.com/wswind/p/16043948.html

# getprop sys.usb.config  # (rndis,adb)
# setprop sys.usb.config diag,serial_cdev,rmnet,adb

# lsmod
# lsmod | grep usbserial

# sudo modprobe -r pl2303
# sudo modprobe -r option
# sudo modprobe -r usb_wwan
# sudo modprobe -r usbserial
sudo rmmod pl2303
sudo rmmod option
sudo rmmod usb_wwan
sudo rmmod usbserial
sleep 1

sudo modprobe usbserial vendor=0x05c6 product=0x9091
# sudo modprobe usbserial vendor=0x2c7c product=0x0800
sleep 1

adb devices
ls /dev/serial/by-id