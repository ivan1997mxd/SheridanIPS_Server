import ast
import csv
import json
import winsound
from statistics import mean
from time import time
from typing import Dict, List
from uuid import uuid4

import xlsxwriter.exceptions
import xlsxwriter
from flask import request

from Algorithms.Combination.Combination import get_combination_function
from Algorithms.NearestNeighbour.NNv4 import get_NNv4, get_NNv4_RSSI
from Offline.MatrixProduction import create_all_matrices_from_rssi_data
from Resources.Objects.Points.Zone import Zone, get_all_zones
from Resources.Objects.Worksheet import Worksheet
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Points.Point import Point
from Resources.Objects.TestData import create_test_data_list

from flask import Flask
from flask_pymongo import PyMongo
from Resources.Scripts.DataFilters.APFilter import rmac_filter_data
from Resources.Scripts.DataFilters.KalmanFilter import kalman_filter_data, kalman_filter_file

app = Flask(__name__)

dates = [
    # ["November 19", "November 20", "November 21"],
    ["April 3", "April 6", "April 8"],
]
times = ["15_00", "18_00", "20_00"]  # readonly
num_combinations = [2, 3]
combination_modes = ["WGT"]  # ["AVG", "WGT", "AB"]
error_modes = ["MAX"]
mongo = PyMongo(app, uri="mongodb://localhost:27017/SheridanIPS")

# 1. Establish file location data.
main_folder = "./Data"
access_point_file_path = "{}/November/Points/Access Points/Access Points Home 5.csv".format(main_folder)
grid_point_file_path = "{}/November/Points/Grid Points/GP_home_5.csv".format(
    main_folder)

centroid_file_path = "{}/November/Points/Centroid Points/Centroid Points_Home_5.csv".format(main_folder)
zone_file_path = "{}/November/Zones/Zones_Home_5.csv".format(main_folder)
sorted_offline_rssi_folder_path = "{}/HOME/5/Offline".format(main_folder)
sorted_online_rssi_folder_path = "{}/HOME/5/Online".format(main_folder)
beacon_folder_Offline_path = "{}/Beacon Data/Kalman-Filtered/Home 5".format(main_folder)
beacon_folder_Online_path = "{}/Beacon Data/RSSI Data/Online/Home 5".format(main_folder)
wifi_folder_Online_path = "{}/Wifi Data/Kalman-Filtered/Home 5".format(main_folder)

# 2. Instantiate "static" objects.
access_points = AccessPoint.create_point_list(file_path=access_point_file_path)
grid_points = GridPoint.create_point_list(file_path=grid_point_file_path, access_points=access_points)
centroids = Centroid.create_point_list(file_path=centroid_file_path, grid_points=grid_points)
zones = get_all_zones(file_path=zone_file_path)
ap_list = [ap.id for ap in access_points]


def tain_data(dates: List[List[str]],
              times: List[str],
              num_combinations: List[int],
              combination_modes: List[str],
              error_modes: List[str]):
    # 3. Start producing Matrices:
    worksheets = list()  # type: List[Worksheet]
    sheet_counter = 1
    num_worksheets = len(combination_modes) * len(error_modes) * len(dates) * len(num_combinations)
    print("There will be a total of {} worksheet{}.".format(num_worksheets, "" if num_worksheets == 1 else "s"))

    for date_subset in dates:

        testing_data = create_test_data_list(access_points=access_points,
                                             zones=zones,
                                             folder_path=sorted_online_rssi_folder_path,
                                             dates=date_subset,
                                             times=times)

        # 5. Get test data.
        training_data = create_test_data_list(access_points=access_points,
                                              zones=zones,
                                              folder_path=sorted_offline_rssi_folder_path,
                                              dates=date_subset,
                                              times=times)

        print("-- Instantiated test lists.")
        print("-- Offline data size: {}".format(len(training_data)))
        print("-- Online data size: {}".format(len(testing_data)))

        for combination_mode in combination_modes:
            combination_method = get_combination_function(combination_mode)

            for error_mode in error_modes:

                # Set error mode:
                NormalizedMatrix.error_mode = error_mode

                for combination in num_combinations:
                    print("Working on sheet {} of {}.".format(sheet_counter, num_worksheets))
                    sheet_counter += 1

                    distributions = create_all_matrices_from_rssi_data(
                        access_points=access_points,
                        centroids=centroids,
                        zones=zones,
                        training_data=training_data,
                        testing_data=testing_data,
                        combination_method=combination_method,
                        num_combinations=combination)

                    # Separate the Tuple retrieved above.
                    probability_distributions = distributions[0]
                    normalized_distributions = distributions[1]
                    resultant_combinations = distributions[2]
                    normalized_resultant_combinations = distributions[3]
                    test_results = distributions[4]

                    worksheets.append(Worksheet(num_combinations=combination,
                                                date_subset=date_subset,
                                                error_mode=error_mode,
                                                combination_mode=combination_mode,
                                                normalized_probability_matrices=normalized_distributions,
                                                normalized_combined_matrices=normalized_resultant_combinations,
                                                test_results=test_results))

                    # Reset the point objects:
                    Point.reset_points()
        # Start saving the workbook:
        excel_start_time = time()
        excel_workbook = xlsxwriter.Workbook("{}/Matrices/Results_5_mix.xlsx".format(main_folder))
        bold = excel_workbook.add_format({'bold': True})
        merge_format = excel_workbook.add_format({'bold': True, 'align': 'center'})

        # Save the key page:
        excel_worksheet = excel_workbook.add_worksheet("Keys")
        Worksheet.save_key_page(excel_worksheet, bold=bold, merge_format=merge_format)

        # Save all other pages:
        problems_saving = list()  # type: List[str]
        for sheet in worksheets:
            try:
                excel_worksheet = excel_workbook.add_worksheet(sheet.tab_title)
            except xlsxwriter.exceptions.DuplicateWorksheetName:

                problem_sheet = sheet.tab_title
                problem_resolution = str(uuid4())[:25]
                problem_description = "Worksheet {} has the same tab-title as another page. It has been replaced with {}."
                problems_saving.append(problem_description.format(problem_sheet, problem_resolution))

                excel_worksheet = excel_workbook.add_worksheet(problem_resolution)

            sheet.save(excel_worksheet, bold=bold, merge_format=merge_format)
        excel_workbook.close()
        excel_end_time = time()

        print("Workbook Write Time: {}s.".format(excel_end_time - excel_start_time))

        if len(problems_saving) > 0:
            print("There were problems saving {} worksheets.".format(len(problems_saving)))
            for problem in problems_saving:
                print(problem)
            print("You can manually change the tab-titles now if desired.")


# region Routes
def filter_data(dict_data):
    data = dict()
    scan_data = [d for d in dict_data if d['BSSID'] in ap_list]
    for d in scan_data:
        data[d['BSSID']] = round(mean(d['RSSIs']))
    return data


# region Routes
@app.route('/')
def hello_world():
    start_time = time()
    tain_data(dates=dates,
              times=times,
              num_combinations=num_combinations,
              combination_modes=combination_modes,
              error_modes=error_modes)
    # create_mix_data_list(wifi_folder_Online_path, beacon_folder_Offline_path, dates[0], times)
    end_time = time()
    process_time = end_time - start_time
    return 'Hello, World! ' + str(process_time)


def get_location(path):
    bssid_rssi_dict = dict()  # type: Dict[AccessPoint, int]
    with open(path, "r", newline='') as csvFile:
        reader = csv.reader(csvFile)
        for scan in reader:
            bssid = scan[0]
            rssis = round(mean([int(x) for x in scan[1:]]))
            for access_point in access_points:
                if bssid == access_point.id:
                    bssid_rssi_dict[access_point] = rssis
    point = get_NNv4_RSSI(centroids, bssid_rssi_dict)
    measured_zone = get_zone(zones, point)
    return measured_zone


@app.route('/map', methods=['PUT'])
def test():
    frequency = 4000
    duration = 1000
    winsound.Beep(frequency, duration)
    dict_data = ast.literal_eval(request.form.getlist('PutData')[0])
    scan_data = dict_data['Scans']['data']
    # data = filter_data(scan_data)
    path = store_data(scan_data)
    # print(data)
    # response = get_location(path)
    return json.dumps("unknown")


def store_data(dict_data):
    dates = ["March 30", "April 3", "April 6", "April 8"]
    floors = ["Home 5", "Home 6"]
    times = ["15_00", "18_00", "20_00"]
    tests = ["Grid Points", "Center of Zone", "Off-Center"]
    grid_points = [i for i in range(1, 21)]
    zones = [i for i in range(1, 7)]

    floor = ""
    while floor not in floors:
        print("What is the floor of this test? Enter the integer.")
        for index, floor in enumerate(floors):
            print("{}. {}".format(str(index + 1), floor))

        # usr_input = 1
        usr_input = int(input()) - 1

        if usr_input < 0 or usr_input > len(floors) - 1:
            floor = ""
            continue

        floor = floors[usr_input]

    # Get date:
    date = ""
    while date not in dates:
        print("What is the date of this test? Enter the integer.")
        for index, date in enumerate(dates):
            print("{}. {}".format(str(index + 1), date))

        # usr_input = 1
        usr_input = int(input()) - 1

        if usr_input < 0 or usr_input > len(dates) - 1:
            date = ""
            continue

        date = dates[usr_input]

    # Get time:
    time = ""
    while time not in times:
        print("What time frame does this test data belong to? Enter the integer.")
        for index, time in enumerate(times):
            print("{}. {}".format(str(index + 1), time))

        # usr_input = 1
        usr_input = int(input()) - 1

        if usr_input < 0 or usr_input > len(times) - 1:
            time = ""
            continue

        time = times[usr_input]

    # Get test type:
    test = ""
    while test not in tests:
        print("What is the test type? Enter the integer.")
        for index, test in enumerate(tests):
            print("{}. {}".format(str(index + 1), test))

        # usr_input = 0
        usr_input = int(input()) - 1

        if usr_input < 0 or usr_input > len(tests) - 1:
            test = ""
            continue

        test = tests[usr_input]

    # Get filename
    filename = ""
    if test == "Grid Points":
        print("What Grid Point does this test belong to?")
        print(grid_points)
        filename = "Grid Point {}.csv".format(input())
    elif test == "Center of Zone":
        print("What Zone did this test belong to?")
        print(zones)
        filename = "Center of Zone {}.csv".format(input())
    else:
        print("What Zone does this test belong to?")
        print(zones)
        filename = "Off-Center {}.csv".format(input())

    # Set folder path:
    folder_path = "./Data/Wifi Data/Raw Data/{}/{}/{}/{}/".format(floor, date, time, test)
    raw_path = "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Raw Data/{}/{}/{}/{}/".format(floor, date, time, test)
    rmac_path = "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Floor-Filtered/{}/{}/{}/{}/".format(floor, date, time,
                                                                                                 test)
    kalman_path = "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/Wifi Data/Kalman-Filtered/{}/{}/{}/{}/".format(floor, date, time,
                                                                                                    test)

    with open(folder_path + filename, mode='w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for data in dict_data:
            list_data = [data['BSSID']]
            for r in data['RSSIs']:
                list_data.append(r)
            writer.writerow(list_data)

    aps = ["c4:09:38:6d:3b:09",
           "f8:ab:05:4f:90:86",
           "18:d6:c7:4d:3b:ec"]

    rmac_filter_data(raw_path + filename, rmac_path + filename, aps)
    kalman_filter_file(rmac_path + filename, kalman_path + filename)

    return kalman_path + filename


if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0', port=5000)
