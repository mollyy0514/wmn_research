import os
import shutil

##### VARIABLES SETTING #####
# root_dir = "/Volumes/mollyT7/"
root_dir = "/home/wmnlab/F/quic_data/"
dates = ["2024-09-03"]
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
                  "sm09": [5218, 5219],
                  "vir0": [5290, 5291],
                  "vir1": [5292, 5293]
                  }
num_devices = len(device_names)
quic_exp_names = [exp_name for exp_name in exp_names if "QUIC" in exp_name]
udp_exp_names = [exp_name for exp_name in exp_names if "UDP" in exp_name]
##### VARIABLES SETTING #####

##### CREATING DERICTORIES #####
def create_directory_structure(root_dir, exp_name, device_names, num_devices, num_experiments):
    for device in device_names:
            device_dir = os.path.join(root_dir, exp_name, device)
            print(device_dir)
            os.makedirs(device_dir, exist_ok=True)
            for i in range(1, num_experiments + 1):
                exp_dir = os.path.join(device_dir, f"#{i:02d}")
                os.makedirs(exp_dir, exist_ok=True)
                sub_dirs = ["raw", "middle", "data", "statistics"]
                for sub_dir in sub_dirs:
                    dir_path = os.path.join(exp_dir, sub_dir)
                    os.makedirs(dir_path, exist_ok=True)

for date in dates:
    date_dir = root_dir + date
    for exp_name in exp_names:
        create_directory_structure(date_dir, exp_name, device_names, num_devices, num_experiments)
print("Directory structure created successfully.")
##### CREATING DERICTORIES #####
        
##### MOVING xml of MOBILEINSIGHT FILES #####
for date in dates:
    date_dir = os.path.join(root_dir, date)
    mi2log_dir = os.path.join(root_dir, date, "mi2log_xml")
    file_list = sorted(os.listdir(mi2log_dir))
    for exp_name in exp_names:
        for i in range(1, num_experiments + 1):
            for device in device_names:
                device_dir = os.path.join(date_dir, exp_name, device)
                raw_dir = os.path.join(device_dir, f"#{i:02d}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                for file in file_list:
                    if file.startswith("diag_log_" + device):
                        src_file = os.path.join(mi2log_dir, file)
                        file_list.remove(file)
                        shutil.copy(src_file, raw_dir)
                        break
##### MOVING xml of MOBILEINSIGHT FILES #####

##### MOVING QUIC SERVER qlog FILES #####
for date in dates:
    date_dir = os.path.join(root_dir, date)
    for exp_name in quic_exp_names:
        server_dir = os.path.join(root_dir, date, "server_qlog")
        file_list = sorted([file for file in os.listdir(server_dir) if file.endswith(".qlog")])
        for i in range(1, num_experiments+1):
            for device in device_names:
                device_dir = os.path.join(date_dir, exp_name, device)
                raw_dir = os.path.join(device_dir, f"#{i:02d}", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                for t in range(2):
                    port_num = file_list[0].split("_")[3]
                    port_num = int(port_num)
                    if port_num in device_to_port[device]:
                        src_file = os.path.join(server_dir, file_list[0])
                        file_list.remove(file_list[0])
                        shutil.move(src_file, raw_dir)
##### MOVING QUIC SERVER qlog FILES #####

##### MOVING QUIC CLIENT qlog FILES #####
for date in dates:
    date_dir = os.path.join(root_dir, date)
    for client in device_names:
        client_dir = os.path.join(root_dir, date, client)
        file_list = sorted([file for file in os.listdir(client_dir) if file.endswith(".qlog")])
        for exp_name in quic_exp_names:
            for i in range(1, num_experiments+1):
                    device_dir = os.path.join(date_dir, exp_name, client)
                    raw_dir = os.path.join(device_dir, f"#{i:02d}", "raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    for t in range(2):
                        port_num = file_list[0].split("_")[3]
                        port_num = int(port_num)
                        if port_num in device_to_port[client]:
                            src_file = os.path.join(client_dir, file_list[0])
                            file_list.remove(file_list[0])
                            shutil.move(src_file, raw_dir)
##### MOVING QUIC CLIENT qlog FILES #####

# ##### MOVING UDP SERVER pcap FILES #####
# for date in dates:
#     date_dir = os.path.join(root_dir, date)
#     server_dir = os.path.join(root_dir, date, "server_pcap")
#     file_list = sorted(os.listdir(server_dir))
#     for exp_name in udp_exp_names:
#         for i in range(1, num_experiments + 1):
#             for device in device_names:
#                 device_dir = os.path.join(date_dir, exp_name, device)
#                 raw_dir = os.path.join(device_dir, f"#{i:02d}", "raw")
#                 os.makedirs(raw_dir, exist_ok=True)
#                 for file in file_list:
#                     if file.startswith("server_pcap_BL_" + device):
#                         src_file = os.path.join(server_dir, file)
#                         file_list.remove(file)
#                         shutil.move(src_file, raw_dir)
#                         break
# ##### MOVING UDP SERVER pcap FILES #####

# ##### MOVING UDP CLIENT pcap FILES #####
# for date in dates:
#     date_dir = os.path.join(root_dir, date)
#     for client in device_names:
#         client_dir = os.path.join(root_dir, date, client, date, "client_pcap")
#         file_list = sorted(os.listdir(client_dir))
#         for exp_name in udp_exp_names:
#             for i in range(1, num_experiments + 1):
#                 device_dir = os.path.join(date_dir, exp_name, client)
#                 raw_dir = os.path.join(device_dir, f"#{i:02d}", "raw")
#                 os.makedirs(raw_dir, exist_ok=True)
#                 for file in file_list:
#                     if file.startswith("client_pcap_BL_" + client):
#                         src_file = os.path.join(client_dir, file)
#                         file_list.remove(file)
#                         shutil.move(src_file, raw_dir)
#                         break
# ##### MOVING UDP CLIENT pcap FILES #####