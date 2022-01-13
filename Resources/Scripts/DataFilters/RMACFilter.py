import math

import numpy as np
from flask import Flask
from flask_pymongo import PyMongo
from statistics import mean

# from .SharedFilterResources import __ScanValue
from typing import List, Dict
import csv

from matplotlib import pyplot as plt
from pykalman import KalmanFilter

app = Flask(__name__)
aps = ["a4:ce:da:58:e1:4f", "78:dd:12:1e:3b:1a", "70:f1:96:86:9f:76", "72:dd:12:1e:3b:18", "62:dd:12:1e:3b:18"]


# def rmac_filter_data(raw_data_file_path: str, rmacs: List[str]) -> Dict[str, __ScanValue]:
#     # raise Exception("This method should not be run unless there is new scan data. The existing data is already sorted.")
#
#     scan_dict = dict()  # type: Dict[str, __ScanValue]
#
#     with open(raw_data_file_path) as csvFile:
#         readCSV = csv.reader(csvFile, delimiter=",")
#
#         for line, scan in enumerate(readCSV):
#
#             if scan[0] not in rmacs:
#                 continue
#
#             key = scan[0] + "-" + scan[1]
#
#             if key in scan_dict.keys():
#                 for rssi in scan[2:]:
#                     scan_dict[key].rssis = int(rssi)
#             else:
#                 scanvalue = __ScanValue(scan[0], scan[1])
#                 for rssi in scan[2:]:
#                     scanvalue.rssis = int(rssi)
#                 scan_dict[key] = scanvalue
#
#     return scan_dict


def statistic():
    ap_list = dict()
    mongo = PyMongo(app, uri="mongodb://localhost:27017/SheridanIPS")
    data = list(mongo.db.random_ble_data.find({}, {'_id': False}))
    print(len(data))
    for d in data:
        for key, value in d.items():
            for ap in value:
                bssid = ap['BSSID']
                rssi = ap['RSSIs']
                if bssid not in ap_list.keys():
                    ap_list[bssid] = 1
                else:
                    ap_list[bssid] += 1
    ap_list = sorted(ap_list.items(), key=lambda kv: kv[1], reverse=True)
    print(ap_list)


def raw_statistic():
    ap_list = [
        "C7:F8:C6:2F:15:BB",
        "FD:82:B0:4C:91:BF",
        "C3:88:F6:29:A9:DE",
        "CE:E4:05:BE:70:CD",
        "EE:AF:B9:2E:2C:33",
        "D2:8C:33:3D:CC:0C"
    ]
    mongo = PyMongo(app, uri="mongodb://localhost:27017/SheridanIPS")
    data = list(mongo.db.random_ble_data.find({}, {'_id': False}))
    for index, d in enumerate(data):
        for key, value in d.items():
            data_set = dict()
            gps = list()
            for ap in value:
                bssid = ap['BSSID']
                if bssid in ap_list:
                    gps.append({"BSSID": bssid, "RSSIs": ap['RSSIs']})
            if len(gps) < 6:
                print(index)
            data_set[key] = gps
            # mongo.db.kalman_data.insert(data_set)
            # print(data_set)


def output():
    data_dict = dict()

    mongo = PyMongo(app, uri="mongodb://localhost:27017/SheridanIPS")
    data = list(mongo.db.ble_data.find({}, {'_id': False}))
    for d in data:
        data_set = dict()
        for key, value in d.items():
            for ap in value:
                # avg = round(mean(ap['RSSIs']))
                data_set[ap['BSSID']] = ap['RSSIs']
            if key in data_dict.keys():
                data_dict[key].append(data_set)
            else:
                data_dict[key] = [data_set]

    # print(data_dict)
    find_avg(data_dict)
    # final_data = []
    # for ssid, rss in data_dict.items():
    #     filtered_rss = filterRSS(rss)
    #     gp = int(ssid[2:])
    #     data_object = {"zone_num": gp, "zone_data": filtered_rss}
    #     final_data.append(data_object)
    # final_data = sorted(final_data, key=lambda kv: kv["zone_num"], reverse=False)
    # print(final_data)


def filterRSS(rss):
    ap_list = [
        "C7:F8:C6:2F:15:BB",
        "FD:82:B0:4C:91:BF",
        "C3:88:F6:29:A9:DE",
        "CE:E4:05:BE:70:CD",
        "EE:AF:B9:2E:2C:33",
        "D2:8C:33:3D:CC:0C"
    ]
    for i in range(len(rss)):
        for ap in ap_list:
            data_list = rss[i][ap]
            filtered_data = calculate(data_list, 50.0, 0.008)
            rss[i][ap] = filtered_data
    return rss


def find_avg(data_dict):
    avg_list = list()
    ap_list = [
        "C7:F8:C6:2F:15:BB",
        "FD:82:B0:4C:91:BF",
        "C3:88:F6:29:A9:DE",
        "CE:E4:05:BE:70:CD",
        "EE:AF:B9:2E:2C:33",
        "D2:8C:33:3D:CC:0C"
    ]
    for gp, data in data_dict.items():
        ap_dict = dict()
        for ap in ap_list:
            data_list = list()
            for d in data:
                for mac, value in d.items():
                    if mac == ap:
                        data_list += value
            filtered_data = calculate(data_list, 50.0, 0.008)
            ap_dict[ap] = round(mean(filtered_data))
        gp = int(gp[2:])
        # if gp >= 25:
        #     gp = gp - 24
        avg_list.append({"zone_num": gp, "zone_data": ap_dict})
    avg_list = sorted(avg_list, key=lambda kv: kv["zone_num"], reverse=False)
    print(avg_list)
    # for ap in ap_list:
    #     gp_list = []
    #     data_list = []
    #     for key, item in data_dict.items():
    #         for i in item:
    #             data_list += i[ap]
    #         # kf = kalman_filter(data_list)
    #         filtered_data = calculate(data_list, 50.0, 0.008)
    #         # gp = int(key[2:])
    #         # if gp >= 25:
    #             plt.figure(3)
    #             plt.plot(data_list)
    #             plt.plot(filtered_data)
    #             plt.show()
    #         gp_list.append({key: round(mean(filtered_data))})
    #         data_list = []
    #     ap_dict[ap] = gp_list
    # print(ap_dict)


def calculate(inputValues: list, initialVariance: float, noise: float):
    kalmanGain = 0.0
    processNoise = noise
    variance_value = initialVariance
    measurementNoise = variance(inputValues)
    mean = inputValues[0]
    filtered_list = []

    for value in inputValues:
        variance_value = variance_value + processNoise
        kalmanGain = variance_value / (variance_value + measurementNoise)
        mean = mean + kalmanGain * (value - mean)
        variance_value = variance_value - (kalmanGain * variance_value)
        filtered_list.append(round(mean))

    return filtered_list


def variance(values: list) -> float:
    sum = 0.0
    average = mean(values)
    for v in values:
        sum += math.pow(v - average, 2)
    return sum / (len(values) - 1)


# raw_statistic()
# statistic()
output()
