import csv
import math
from os import listdir
from os.path import isfile, join
from typing import List, Dict
from statistics import mean
from pykalman import KalmanFilter
import numpy as np


def kalman_filter_data(raw_data_file_path: str,
                       filtered_data_file_path: str,
                       input_variance: float = 50.0,
                       noise: float = 0.008):
    # dates = ["November 19", "November 20", "November 21", "November 23"]
    # times = ["15_00", "17_00", "19_00"]
    dates = ["March 30"]
    times = ["15_00"]
    point_types = ["Grid Points"]

    for date in dates:

        for time in times:

            for point_type in point_types:

                raw_data_source_path = "{}/{}/{}/{}/".format(raw_data_file_path, date, time, point_type)
                filtered_data_path = "{}/{}/{}/{}/".format(filtered_data_file_path, date, time, point_type)
                data_files = [f for f in listdir(raw_data_source_path) if isfile(join(raw_data_source_path, f))]
                for file in data_files:
                    bssid_rssi_dict = dict()  # type: Dict[str, List[int]]
                    with open(raw_data_source_path + file, "r", newline='') as csvFile:
                        reader = csv.reader(csvFile)
                        for scan in reader:
                            bssid = scan[0]
                            rssis = [int(x) for x in scan[1:]]
                            if len(rssis) > 1:
                                filtered_rssis = calculate(rssis, input_variance, noise)
                                bssid_rssi_dict[bssid] = filtered_rssis
                            else:
                                bssid_rssi_dict[bssid] = rssis
                    with open(filtered_data_path + file, mode='w', newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        for key, values in bssid_rssi_dict.items():
                            list_data = [key]
                            for v in values:
                                list_data.append(v)
                            writer.writerow(list_data)


def kalman_filter_file(raw_data_file_path: str,
                       filtered_data_file_path: str,
                       input_variance: float = 50.0,
                       noise: float = 0.008):
    bssid_rssi_dict = dict()  # type: Dict[str, List[int]]
    with open(raw_data_file_path, "r", newline='') as csvFile:
        reader = csv.reader(csvFile)
        for scan in reader:
            bssid = scan[0]
            rssis = [int(x) for x in scan[1:]]
            if len(rssis) > 1:
                filtered_rssis = calculate(rssis, input_variance, noise)
                bssid_rssi_dict[bssid] = filtered_rssis
            else:
                bssid_rssi_dict[bssid] = rssis
    with open(filtered_data_file_path, mode='w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, values in bssid_rssi_dict.items():
            list_data = [key]
            for v in values:
                list_data.append(v)
            writer.writerow(list_data)


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


def kalman_filter(value_list: list):
    measurements = np.asarray(value_list)
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=measurements[0],
                      initial_state_covariance=1,
                      observation_covariance=8,
                      transition_covariance=9)  # 0.01)
    state_means, state_covariances = kf.filter(measurements)
    state_std = np.sqrt(state_covariances[:, 0])


# kalman_filter_file("D:/Code/SheridanIPS_Server/Data/RMAC-Filtered/April 3/18_00/Center of Zone/Center of Zone 5.csv", "D:/Code/SheridanIPS_Server/Data/Kalman Filter Values/April 3/18_00/Center of Zone/Center of Zone 5.csv")
# kalman_filter_data("C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Floor-Filtered/Home 5/", "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Kalman-Filtered/Home 5/")
# public class Kalman {
#
# /* Complete calculation of Kalman Filter */
# public static Double kalman (ArrayList<Double> inputValues, double initialVariance, double noise){
#     return calculate(inputValues, initialVariance, noise);
# }
#
# /* Calculation of Kalman Filter using default values for wireless Access Points data acquisition */
# public static Double kalman (ArrayList<Double> inputValues){
#     return calculate(inputValues, 50.0, 0.008);
# }
#
# /* Calculation of arithmetic mean */
# public static Double mean (ArrayList<Double> inputValues){
#     return StatUtils.mean(inputValues);
# }
#
#
# /*This method is the responsible for calculating the value refined with Kalman Filter */
# private static Double calculate(ArrayList<Double> inputValues, double initialVariance, double noise){
#     Double kalmanGain;
#     Double variance = initialVariance;
#     Double processNoise = noise;
#     Double measurementNoise = StatUtils.variance(inputValues);
#     Double mean = inputValues.get(0);
#
#     for (Double value : inputValues){
#         variance = variance + processNoise;
#         kalmanGain = variance/((variance+measurementNoise));
#         mean = mean + kalmanGain*(value - mean);
#         variance = variance - (kalmanGain*variance);
#     }
#
#     return mean;
# }
#
# public class StatUtils {
#
# static Double variance (ArrayList<Double> values){
#     Double sum = 0.0;
#     Double mean = mean(values);
#     for(double num : values){
#         sum += Math.pow(num - mean , 2);
#     }
#     return sum/(values.size()-1);
# }
#
# static Double mean (ArrayList<Double> values){
#     return sum(values)/values.size();
# }
#
# private static Double sum (ArrayList<Double> values){
#     Double sum = 0.0;
#     for (Double num : values){
#         sum+=num;
#     }
#     return sum;
# }
# }
