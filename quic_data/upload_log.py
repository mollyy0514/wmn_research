"""
TODO: 
    1. Implement scp command.
USAGE:
    1. add -c to the command line for the client to execute
"""
import os
import sys
import shutil
import argparse
import subprocess

##### VARIABLES SETTING #####
# root_dir = "/Volumes/mollyT7/"
root_dir = "/home/wmnlab/F/quic_data"

target_date = "2024-09-03"
exp_names = ["QUIC-bbr"]
device_names = ["sm01", "sm02"]
num_experiments = 4

device_to_port = {"sm00": [5200, 5201], 
                  "sm01": [5202, 5203], 
                  "sm02": [5204, 5205],
                  "sm03": [5206, 5207],
                  "sm04": [5208, 5209],
                  "sm05": [5210, 5211],
                  "sm06": [5212, 5213],
                  "sm07": [5214, 5215],
                  "sm08": [5216, 5217],
                  "sm09": [5218, 5219],}

serial_to_device = {"R5CR20FDXHK":"sm00",
                    "R5CR30P9Z8Y":"sm01",
                    "R5CRA1GCHFV":"sm02",
                    "R5CRA1JYYQJ":"sm03",
                    "R5CRA1EV0XH":"sm04",
                    "R5CRA1GBLAZ":"sm05",
                    "R5CRA1ESYWM":"sm06",
                    "R5CRA1ET22M":"sm07",
                    "R5CRA1D23QK":"sm08",
                    "R5CRA2EGJ5X":"sm09",
                    "R5CRA1ET5KB":"sm10",
                    "R5CRA1D2MRJ":"sm11",}

qlog_phone_folder = os.path.join("/sdcard/experiment_log", target_date)
sync_phone_folder = os.path.join("/sdcard/experiment_log", target_date, "sync")
server_folder = os.path.join("/home/wmnlab/temp", target_date)
dest_computer_folder = os.path.join("/home/wmnlab/Desktop/experiment_log", target_date)

parser = argparse.ArgumentParser()
parser.add_argument('-h', "--host", help="server ip address", default="wmnlab@140.112.20.183")
parser.add_argument('-s', "--server", action='store_true', help="For the server to upload files.")
parser.add_argument('-c', "--client", action='store_true', help="For the client to upload files.")
parser.add_argument('-p', "--password", type=str, help="password for execution")
args = parser.parse_args()
##### VARIABLES SETTING #####

##### GLOBAL VARIABLE & FUNCTIONS #####
# def run_scp(directory, destination, password, port=7777):
#     """Runs the SCP command to copy a directory to a remote destination."""
#     try:
#         command = f"sshpass -p '{password}' scp -r -P {port} {directory} {destination}"
#         subprocess.run(command, shell=True, check=True)
#         print(f"Successfully copied {directory} to {destination} on port {port}.")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to copy {directory} to {destination}. Error: {e}")
##### GLOBAL VARIABLE & FUNCTIONS #####
    
##### CLIENT UPLOAD #####
if args.client:
    ##### READ DEVICES #####
    # Read devices connected to computer
    adb_devices_command = "adb devices"
    output = os.popen(adb_devices_command).read()

    # check the devices
    lines = output.strip().split("\n")[1:]
    serial_numbers = {}
    unauthorized_serial_numbers = {}
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            serial_number, status = parts[:2]
            if status == "device":
                serial_numbers[serial_to_device[serial_number]] = serial_number
            elif status == "unauthorized":
                unauthorized_serial_numbers[serial_to_device[serial_number]] = serial_number
    # sort 
    serial_numbers = dict(sorted(serial_numbers.items()))
    unauthorized_serial_numbers = dict(sorted(unauthorized_serial_numbers.items()))
    print("Connected devices:")
    print(serial_numbers)
    print("Unauthorized devices:")
    print(unauthorized_serial_numbers)
    if len(unauthorized_serial_numbers) != 0:
        print("Error: There's unauthorized devices.")
        sys.exit(1)
    ##### READ DEVICES #####

    ##### UPLOAD FILES #####
    # copy the files to the connected computer
    def pull_files(files, serial, dest_path):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for filename in files:
            adb_command = f"adb -s {serial} pull {filename} {dest_path}"
            os.system(adb_command)

    # copy exp logs to the computer
    for dev, serial in serial_numbers.items():
        adb_ls_command = f"adb -s {serial} shell ls {qlog_phone_folder}"
        output = os.popen(adb_ls_command).read()
        phone_files = output.split()
        qlog_files = [filename for filename in phone_files if filename.startswith("log") and filename.endswith(".qlog")]
        time_files = [filename for filename in phone_files if filename.startswith("time") and filename.endswith(".txt")]
        tls_files = [filename for filename in phone_files if filename.startswith("tls_key") and filename.endswith(".log")]
        pull_files(qlog_files, serial, os.join(dest_computer_folder, dev, "client_qlog"))
        pull_files(time_files, serial, os.join(dest_computer_folder, dev, "time_file"))
        pull_files(tls_files, serial, os.join(dest_computer_folder, dev, "tls_key"))

    # copy sync files to the computer
    for dev, serial in serial_numbers.items():
        adb_ls_command = f"adb -s {serial} shell ls {sync_phone_folder}"
        output = os.popen(adb_ls_command).read()
        phone_files = output.split()
        sync_files = [filename for filename in phone_files if filename.startswith("time_sync") and filename.endswith(".json")]
        pull_files(sync_files, serial, dest_computer_folder)
    ##### UPLOAD FILES #####

    # ##### SCP FILES #####
    # run_scp(dest_computer_folder, args.password, args.host+":"+root_dir)
    # ##### SCP FILES #####
##### CLIENT UPLOAD #####

##### SERVER UPLOAD #####
# elif args.server:
    # ##### SCP FILES #####
    # run_scp(os.join(server_folder, "server_qlog"), args.password, args.host+":"+root_dir)
    # run_scp(os.join(server_folder, "time_file"), args.password, args.host+":"+root_dir)
    # run_scp(os.join(server_folder, "tls_key"), args.password, args.host+":"+root_dir)
    # ##### SCP FILES #####
##### SERVER UPLOAD #####

else:
    print("Error: You must provide either -s or -c as an argument.")
    sys.exit(1)