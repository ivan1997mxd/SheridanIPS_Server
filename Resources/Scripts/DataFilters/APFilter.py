import ast
import csv
import os
from os import listdir
from os.path import isfile, join
from statistics import mean
from typing import List, Dict

from kombu.utils import nested

from Resources.Objects.Points import AccessPoint


def create_list(folder_path: str, create_path: str, dates: List[str],
                times: List[str]):
    list_ap = list()
    for date in dates:
        for time in times:

            sub_folder = "{}/{}/{}/Grid Points/".format(folder_path, date, time)

            data_files = [f for f in listdir(sub_folder) if isfile(join(sub_folder, f))]

            for file in data_files:
                detail_list = list()
                gp = int(file[11:-4])
                detail_list.append(gp)
                with open(sub_folder + file, "r", newline='') as csvFile:
                    reader = csv.reader(csvFile)
                    for scan in reader:
                        bssid = scan[0]
                        rssis = [int(x) for x in scan[1:]]
                        value = round(mean(rssis))
                        detail_list.append(bssid)
                        detail_list.append(value)
                list_ap.append(detail_list)
    with open(create_path, mode='w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for data in list_ap:
            writer.writerow(data)


def ap_list(folder_path: str, dates: List[str],
            times: List[str]) -> list:
    list_ap = list()

    ap_dict = dict()  # type: Dict[str, int]

    for date in dates:
        for time in times:

            sub_folder = "{}/{}/{}/Grid Points/".format(folder_path, date, time)

            data_files = [f for f in listdir(sub_folder) if isfile(join(sub_folder, f))]

            data_list = list()

            for file in data_files:
                ap_details = dict()
                detail_list = list()
                data = dict()  # type: Dict[str, int]
                gp = int(file[11:-4])
                with open(sub_folder + file, "r", newline='') as csvFile:
                    reader = csv.reader(csvFile)
                    for scan in reader:
                        bssid = scan[0]
                        rssis = [int(x) for x in scan[1:]]
                        row_list = [bssid]
                        detail_list.append(row_list)
                        if len(rssis) > 28:
                            value = round(mean(rssis))
                            data[bssid] = value
                    sorted_data = sorted(data.items(), key=lambda d: d[1], reverse=True)
                    for key, value in sorted_data:
                        if key in ap_dict.keys():
                            ap_dict[key] += 1
                        else:
                            ap_dict[key] = 1
                    # sorted_data = sorted_data[:6]
                ap_details["id"] = gp
                ap_details["data"] = sorted_data

                list_ap.append(ap_details)
    sort_ap_dict = sorted(ap_dict.items(), key=lambda item: item[1], reverse=True)
    print(sort_ap_dict)
    sort_ap_dict = sort_ap_dict[:8]
    return list_ap


def ap_filter(folder_path: str, dates: List[str],
              times: List[str]) -> list:
    list_ap = list()
    aps = ["c4:09:38:6d:3b:09",
           "f8:ab:05:4f:90:86",
           "18:d6:c7:4d:3b:ec",
           "18:90:d8:dc:2b:2e",
           "34:8a:ae:6b:5e:6e"]

    for date in dates:
        for time in times:

            sub_folder = "{}/{}/{}/Grid Points/".format(folder_path, date, time)

            data_files = [f for f in listdir(sub_folder) if isfile(join(sub_folder, f))]

            for file in data_files:
                ap_details = dict()
                data = dict()  # type: Dict[str, int]
                gp = int(file[11:-4])
                with open(sub_folder + file, "r", newline='') as csvFile:
                    reader = csv.reader(csvFile)
                    for scan in reader:
                        bssid = scan[0]
                        if bssid in aps:
                            rssis = [int(x) for x in scan[1:]]
                            value = round(mean(rssis))
                            data[bssid] = value

                    sorted_data = sorted(data.items(), key=lambda d: d[1], reverse=True)
                    # sorted_data = sorted_data[:6]
                ap_details["id"] = gp
                ap_details["data"] = sorted_data

                list_ap.append(ap_details)
    print(list_ap)
    return list_ap


def rmac_filter_data(raw_data_file_path: str, filtered_data_file_path: str, aps: list):
    # data_files = [f for f in listdir(raw_data_file_path) if isfile(join(raw_data_file_path, f))]
    # for file in data_files:
    bssid_rssi_dict = dict()  # type: Dict[str, List[int]]
    with open(raw_data_file_path, "r", newline='') as csvFile:
        reader = csv.reader(csvFile)
        for scan in reader:
            bssid = scan[0]
            rssis = [int(x) for x in scan[1:]]
            bssid_rssi_dict[bssid] = rssis
    with open(filtered_data_file_path, mode='w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, values in bssid_rssi_dict.items():
            if key in aps:
                list_data = [key]
                for v in values:
                    list_data.append(v)
                writer.writerow(list_data)
            else:
                continue


def filter_bssid(raw_data_file_path: str, filtered_data_file_path: str):
    bssids = ["c4:09:38:6d:3b:09",
              "f8:ab:05:4f:90:86",
              "18:d6:c7:4d:3b:ec"]

    data_files = [f for f in listdir(raw_data_file_path) if isfile(join(raw_data_file_path, f))]
    for file in data_files:
        ap_details = list()
        with open(raw_data_file_path + file, "r", newline='') as csvFile:
            reader = csv.reader(csvFile)
            for scan in reader:
                bssid = scan[0]
                rssis = [int(x) for x in scan[1:]]
                if bssid in bssids:
                    row_list = [bssid]
                    for v in rssis:
                        row_list.append(v)
                    ap_details.append(row_list)
        if not os.path.exists(filtered_data_file_path):
            os.mkdir(filtered_data_file_path)
        with open(filtered_data_file_path + file, mode='w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in ap_details:
                writer.writerow(row)


def avg_ap(raw_data_file_path: str):
    dates = ["April 3", "April 6", "April 8"]
    times = ["15_00", "18_00", "20_00"]
    point_types = ["Center of Zone"]
    list_data = list()

    for date in dates:

        for time in times:

            for point_type in point_types:
                raw_data_source_path = "{}/{}/{}/{}/".format(raw_data_file_path, date, time, point_type)
                data_files = [f for f in listdir(raw_data_source_path) if isfile(join(raw_data_source_path, f))]
                list_gp = list()
                for file in data_files:
                    ap_details = dict()  # type: Dict[str, int]
                    gp = int(file[15:-4])
                    ap_details["id"] = gp
                    with open(raw_data_source_path + file, "r", newline='') as csvFile:
                        reader = csv.reader(csvFile)
                        for scan in reader:
                            bssid = scan[0]
                            rssis = [int(x) for x in scan[1:]]
                            value = round(mean(rssis))
                            ap_details[bssid] = value
                    list_gp.append(ap_details)
                list_data.append(list_gp)
    print(list_data)
    new_dict = dict()
    new_list = list()
    for data in list_data:
        d = data[5]
        data_list = [v for v in d.values() if v != 6]
        new_list.append(data_list)
    print(new_list)
    avg = list()
    for d in new_list:
        avg.append(d[6])
    print(avg)
    answer = my_average_main(avg)
    print(answer)

    return list_data


def my_average_main(data_list):
    if len(data_list) == 0:
        return 0
    if len(data_list) > 2:
        data_list.remove(min(data_list))
        data_list.remove(max(data_list))
        average_data = float(sum(data_list)) / len(data_list)
        return average_data
    elif len(data_list) <= 2:
        average_data = float(sum(data_list)) / len(data_list)
        return average_data


def create_mix_data_list(folder_path_wifi: str,
                         folder_path_beacon: str,
                         dates: List[str],
                         times: List[str]):
    for date in dates:
        for time in times:

            sub_folder_wifi = "{}/{}/{}/Center of Zone/".format(folder_path_wifi, date, time)

            data_files_wifi = [f for f in listdir(sub_folder_wifi) if isfile(join(sub_folder_wifi, f))]

            sub_folder_beacon = "{}/{}/{}/Center of Zone/".format(folder_path_beacon, date, time)

            data_files_beacon = [f for f in listdir(sub_folder_beacon) if isfile(join(sub_folder_beacon, f))]

            for file in data_files_wifi:
                if file in data_files_beacon:
                    with nested(open(sub_folder_wifi + file, mode='a', newline=""),
                                open(sub_folder_beacon + file, mode='r', newline="")) as (
                            new_file, old_file):
                        old_file_reader = csv.reader(old_file)
                        new_file_writer = csv.writer(new_file)
                        for scan in old_file_reader:
                            new_file_writer.writerow(scan)


def csv_to_json(raw_data_file_path: str, dates: List[str], times: List[str], point_types: List[str]):
    json_list = list()
    for date in dates:

        for time in times:

            for point_type in point_types:
                raw_data_source_path = "{}/{}/{}/{}/".format(raw_data_file_path, date, time, point_type)
                data_files = [f for f in listdir(raw_data_source_path) if isfile(join(raw_data_source_path, f))]
                for file in data_files:
                    with open(raw_data_source_path + file, "r", newline='') as csvFile:
                        reader = csv.reader(csvFile)
                        data = dict()
                        for scan in reader:
                            bssid = scan[0]
                            rssis = [int(x) for x in scan[1:]]
                            data[bssid] = rssis
                    gp = int(file[15:-4])
                    found = False
                    for json in json_list:
                        if gp == json["zone_num"]:
                            json["zone_data"].append(data)
                            found = True
                    if not found:
                        data_object = dict()
                        data_object["zone_num"] = gp
                        data_object["zone_data"] = [data]
                        json_list.append(data_object)
    print(json_list)


csv_to_json("E:/Code/SheridanIPS_Server/Data/NewData/Home 6", ["April 3", "April 6", "April 8"],
            ["15_00", "18_00", "20_00"], ["Center of Zone"])

# csv_to_json("D:/Code/SheridanIPS_Server/Data/SCAET/1", ["November 19", "November 20", "November 21", "November 23"],
#             ["15_00", "18_00", "20_00"], ["Center of Zone"])

# create_mix_data_list("E:/Code/SheridanIPS_Server/Data/NewData/Home 6",
#                      "E:/Code/SheridanIPS_Server/Data/Beacon Data/Raw Data/Home 6",
#                      ["April 3", "April 6", "April 8"], ["15_00", "18_00", "20_00"])

# create_list("C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Kalman-Filtered/Home 5",
#             "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/November/Points/Grid Points/GP_home_5_bk.csv",
#             ["March 30"],
#             ["15_00"])

# ap_list("C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Raw Data/Home 5", ["March 30"],
#         ["15_00"])
# filter_bssid(
#     "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Raw Data/Home 5/March 30/15_00/Grid Points/",
#     "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Floor-Filtered/Home 5/March 30/15_00/Grid Points/")
# ap_filter("D:/Code/SheridanIPS_Server/Data/Raw RSSI Values", ["April 3"], ["18_00"])
# avg_ap("C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Kalman-Filtered/Home 5")

# date = ["April 3", "April 6", "April 8"]
# time = ["15_00", "18_00", "20_00"]
# zone = ["Center of Zone 1",
#         "Center of Zone 2",
#         "Center of Zone 3",
#         "Center of Zone 4",
#         "Center of Zone 5",
#         "Center of Zone 6"]
# for d in date:
#     for t in time:
#         for z in zone:
#             rmac_filter_data(
#                 "E:/Code/SheridanIPS_Server/Data/Raw RSSI Values/{}/{}/Center of Zone/{}.csv".format(d, t, z),
#                 "E:/Code/SheridanIPS_Server/Data/NewData/Home 6/{}/{}/Center of Zone/{}.csv".format(d, t, z),
#                 ["c4:09:38:6d:3b:09",
#                  "f8:ab:05:4f:90:86",
#                  "18:d6:c7:4d:3b:ec",
#                  "18:90:d8:dc:2b:2e",
#                  "34:8a:ae:6b:5e:6e"])
