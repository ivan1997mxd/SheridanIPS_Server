import ast
import json
import os
import shutil
from datetime import timedelta, datetime
from random import choice
from urllib import request

import cv2
import pandas as pd
import xlrd
from flask import Flask
from flask_pymongo import PyMongo
from flask import Flask, render_template, flash, redirect, url_for, make_response
from flask import request

from Resources.Objects.Matrices.CombinedDistribution import CombinedMatrix
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix, sort_matrices
from Algorithms.Combination.Combination import get_combination_function
from Resources.Objects.Matrices.ProbabilityDistribution import ProbabilityMatrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint import GridPoint
from Resources.Objects.Points.Centroid import Centroid
from Offline.MatrixProduction import create_all_matrices_from_rssi_data, get_NNv4
from Resources.Objects.TestData import create_test_data_list, TestResult, Sample
from Resources.Objects.Zone import get_all_zones, Zone, get_zone
from Resources.Objects.Points.Point import Point
from Resources.Objects.Worksheet import Worksheet
from Algorithms.Combination.AdaptiveBoosting import create_matrices
from typing import List, Tuple, Dict, Callable
from time import time
from uuid import uuid4
import xlsxwriter.exceptions
import xlsxwriter

ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# Create the application for use
app = Flask(__name__)
app.secret_key = 'sheridanILS'
mongo = PyMongo(app, uri="mongodb://localhost:27017/SheridanIPS")
app.send_file_max_age_default = timedelta(seconds=1)
x = 0
y = 0

# Globals to set parameters for Matrix Production
dates = [
    ["November 19", "November 20", "November 21"],
]
times = ["15_00", "17_00", "19_00"]  # readonly
num_combinations = [2]
combination_modes = ["WGT"]  # ["AVG", "WGT", "AB"]
error_modes = ["MAX"]

# 1. Establish file location data.
main_folder = "./Data/November"
access_point_file_path = "{}/Points/Access Points/Access Points.csv".format(main_folder)
grid_point_file_path = "{}/Points/Grid Points/November 19 - November 20 - November 21 - November 23 Grid Points.csv".format(
    main_folder)
centroid_file_path = "{}/Points/Centroid Points/Centroid Points.csv".format(main_folder)
zone_file_path = "{}/Zones/Zones.csv".format(main_folder)
sorted_offline_rssi_folder_path = "{}/RSSI Data/Test Data/Offline/".format(main_folder)
sorted_online_rssi_folder_path = "{}/RSSI Data/Test Data/Online/".format(main_folder)

# 2. Instantiate "static" objects.
access_points = AccessPoint.create_point_list(file_path=access_point_file_path)
grid_points = GridPoint.create_point_list(file_path=grid_point_file_path, access_points=access_points)
centroids = Centroid.create_point_list(file_path=centroid_file_path, grid_points=grid_points)
zones = get_all_zones(file_path=zone_file_path)


def train_data(dates: List[List[str]],
               times: List[str],
               num_combinations: List[int],
               combination_modes: List[str],
               error_modes: List[str]) -> NormalizedMatrix:
    # 3. Start producing Matrices:
    worksheets = list()  # type: List[Worksheet]
    sheet_counter = 1
    num_worksheets = len(combination_modes) * len(error_modes) * len(dates) * len(num_combinations)
    print("There will be a total of {} worksheet{}.".format(num_worksheets, "" if num_worksheets == 1 else "s"))

    # 4. Start with dates to reduce the number of times test data needs to be read from the CSV files.
    for date_subset in dates:

        # 5. Get test data.
        training_data = create_test_data_list(access_points=access_points,
                                              zones=zones,
                                              folder_path=sorted_offline_rssi_folder_path,
                                              dates=date_subset,
                                              times=times)

        testing_data = create_test_data_list(access_points=access_points,
                                             zones=zones,
                                             folder_path=sorted_online_rssi_folder_path,
                                             dates=date_subset,
                                             times=times)

        print("-- Instantiated test lists.")
        print("-- Offline data size: {}".format(len(training_data)))
        print("-- Online data size: {}".format(len(testing_data)))

        for combination_mode in combination_modes:

            # Set combination mode:
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

                    sort_matrices(matrix_list=distributions[3])

                    return distributions[3][0]


def get_position(normalized: NormalizedMatrix,
                 centroids: List[Centroid],
                 zones: List[Zone],
                 testing_data: Sample,
                 combination_method: Callable) -> Tuple[Zone, Zone]:
    combine_vectors = combination_method

    resultant = normalized.parent_matrix

    # JC-01 used the measured/reported zone instead of the actual zone to the algorithm.
    vectors = list()  # type: List[Dict[Zone, float]]
    answers = list()
    for distribution in resultant.normalizations:
        ap_rssi_dict = testing_data.get_ap_rssi_dict(*distribution.access_points)

        coord = get_NNv4(centroid_points=centroids, rssis=ap_rssi_dict)
        zone = get_zone(zones=zones, co_ordinate=coord)

        vector = distribution.get_vector(zone)
        vectors.append(vector)
        answers.append(zone)

    NormalizedMatrix.theAnswer = testing_data.answer  # JC-01 - used to pass the true answer around for run-time validation - used by dbg_combine_vector

    resultant_vector = combine_vectors(answers, *vectors)
    measured_zone_1 = max(resultant_vector, key=resultant_vector.get)
    resultant_vector[measured_zone_1] = 0.0
    measured_zone_2 = max(resultant_vector, key=resultant_vector.get)

    return measured_zone_1, measured_zone_2


def show_position(img, x, y, name):
    cv2.rectangle(img, (x - 50, y - 50), (x + 50, y + 50), (255, 0, 0), 2)
    cv2.putText(img, name, (x - 30, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.circle(img, (x, y), 5, (0, 255, 0), thickness=-1)
    # rx = random.randrange(x - 60, x + 60, 1)
    # ry = random.randrange(y - 60, y + 60, 1)
    # cv2.circle(img, (rx, ry), 5, (255, 0, 0), thickness=-1)


@app.route('/login', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form.get('form-username')
        password = request.form.get('form-password')
        user = mongo.db.Users.find({'username': username, 'password': password}).count()
        if user == 0:
            print("User Not Exist")
        else:
            print(username + "found")
            return redirect(url_for('home'))
    return render_template('login.html')


@app.route("/setting", methods=["Get"])
def setting():
    return render_template("setting.html")


@app.route("/home", methods=["Get"])
def home():
    return render_template("home.html")


@app.route("/location", methods=["Get", "Post"])
def location():
    basePath = os.path.dirname(__file__)
    src = os.path.join(basePath, 'static/img', 'test.jpg')
    shutil.copyfile(os.path.join(basePath, 'static/img', 'test.jpg'), os.path.join(basePath, 'static/img', 'test1.jpg'))
    return redirect(url_for("update"))


@app.route("/wifi", methods=['GET', 'POST'])
def wifi():
    wifi = mongo.db.WifiScans.find()[0]
    print(wifi['Scans'][0])
    time = datetime.fromtimestamp(wifi['Time'] / 1e3).strftime("%m %d, %Y, %H:%M:%S")
    return render_template('wifi.html', wifi=wifi['Scans'][0], time=time)


@app.route('/training')
def main():
    combination_method = get_combination_function(combination_modes[0])
    data = create_test_data_list(access_points=access_points,
                                 zones=zones,
                                 folder_path=sorted_online_rssi_folder_path,
                                 dates=dates[0],
                                 times=times)
    sample_data = choice(data)
    dataList = train_data(dates=dates,
                          times=times,
                          num_combinations=num_combinations,
                          combination_modes=combination_modes,
                          error_modes=error_modes)

    measured_zone = get_position(normalized=dataList,
                                 centroids=centroids,
                                 zones=zones,
                                 testing_data=sample_data,
                                 combination_method=combination_method)
    return 'the actual zone is ' + str(Sample.answer) + ', the measured zone is ' + str(measured_zone)


@app.route('/update', methods=['Get', 'PUT'])
def update():
    basePath = os.path.dirname(__file__)
    if request.method == 'PUT':
        global time
        # TODO: Perhaps store logs of what has occurred - wiping sensitive data?
        print(request.form.to_dict())
        dict_data = ast.literal_eval(request.form.getlist('PutData')[0])  # Gets the actual JSON data that was sent.
        print(dict_data)
        # new_time = dict_data["Time"]
        # print(new_time - time)
        # time = new_time
        # return str((200, "OK"))
        response = get_position(dict_data)
        print(str(response))
        zone = response["closest_centroid"]
        image = cv2.imread("static/img/test.jpg")
        if zone == 11:
            show_position(image, 170, 300, "Chen")
        elif zone == 10:
            show_position(image, 270, 300, "Chen")
        elif zone == 9:
            show_position(image, 370, 300, "Chen")
        elif zone == 8:
            show_position(image, 470, 300, "Chen")
        elif zone == 7:
            show_position(image, 570, 300, "Chen")
        cv2.imwrite(os.path.join(basePath, 'static/img', 'test1.jpg'), image)
        return json.dumps(response)  # JSON-ifies the dictionary retrieved.

    # img = cv2.imread("static/img/S144_Blank.png")
    # add_line(img)
    # cv2.imwrite(os.path.join(basePath, 'static/img', 'test.jpg'), img)
    return render_template("location.html", img='./img/test1.jpg')


# region Main
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
# endregion
