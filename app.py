import ast
import json
import os
import shutil
from datetime import timedelta, datetime
from math import floor
from random import choice, random
from urllib import request
from concurrent.futures import ThreadPoolExecutor
import cv2
import pandas as pd
import xlrd
from flask import Flask
from flask_pymongo import PyMongo
from flask import Flask, render_template, flash, redirect, url_for, make_response
from flask import request
from celery import Celery
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
from flask_socketio import SocketIO, emit
from Resources.Objects.Worksheet import Worksheet
from Algorithms.Combination.AdaptiveBoosting import create_matrices
from typing import List, Tuple, Dict, Callable
import time
from uuid import uuid4
import xlsxwriter.exceptions
import xlsxwriter

ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}
executor = ThreadPoolExecutor(2)
basePath = os.path.dirname(__file__)


# secondsSinceEpoch = time.time()
# timeObj = time.localtime(secondsSinceEpoch)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# Create the application for use
app = Flask(__name__)
socketio = SocketIO()
socketio.init_app(app)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
app.secret_key = 'sheridanILS'
name_space = '/test'
mongo = PyMongo(app, uri="mongodb://localhost:27017/SheridanIPS")
app.send_file_max_age_default = timedelta(seconds=1)
x = 0
y = 0

# Globals to set parameters for Matrix Production
dates = [
    ["November 19", "November 20", "November 21"],
]
times = ["15_00", "17_00", "19_00"]  # readonly
num_combinations = [3]
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

normalized_matrices = List[NormalizedMatrix]
test_results = Dict[NormalizedMatrix, TestResult]
zone_data = []
location_data = []


@celery.task
def train_data(dates: List[List[str]],
               times: List[str],
               num_combinations: List[int],
               combination_modes: List[str],
               error_modes: List[str]):
    global normalized_matrices
    global test_results
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
                    normalized_matrices = distributions[3]
                    test_results = distributions[4]


def get_test_result(normalized_combined_matrices: List[NormalizedMatrix],
                    test_result: Dict[NormalizedMatrix, TestResult]) -> List:
    ap_results = []
    for index, distribution in enumerate(normalized_combined_matrices):
        result = test_result[distribution]
        AP_Combination = distribution.csv_list[0][0][26:]
        Zones = []
        BaseError1 = []
        BaseError2 = []
        Overall = []
        for zone, zone_results in result.answer_details.items():
            Zones.append(str(zone))
            BaseError1.append(zone_results["base_zone_error_1"] / zone_results["times_tested"])
            BaseError2.append(
                zone_results["base_zone_error_2"] / (zone_results["times_tested"] - zone_results["times_correct"]))
            Overall.append((zone_results["times_correct"] + zone_results["times_2nd_correct"]) / zone_results[
                "times_tested"])
        ap_result = {"AP": AP_Combination,
                     "BaseError1": BaseError1,
                     "BaseError2": BaseError2,
                     "FirstAverage": sum(BaseError1) / len(BaseError1),
                     "SecondAverage": sum(BaseError2) / len(BaseError2),
                     "Overall": sum(Overall) / len(Overall)}
        ap_results.append(ap_result)

    return ap_results


def total_dict(dict):
    sum = 0.0
    for key, value in dict.items():
        sum += value
    return sum


def get_position(normalized: NormalizedMatrix,
                 centroids: List[Centroid],
                 zones: List[Zone],
                 testing_data: Sample,
                 combination_method: Callable) -> Tuple[Zone, Zone, List[float]]:
    combine_vectors = combination_method

    resultant = normalized.parent_matrix

    # JC-01 used the measured/reported zone instead of the actual zone to the algorithm.
    vectors = list()  # type: List[Dict[Zone, float]]
    answers = list()
    for distribution in resultant.normalizations:
        ap_rssi_dict = testing_data.get_ap_rssi_dict(*distribution.access_points)
        print(str(ap_rssi_dict))
        coord = get_NNv4(centroid_points=centroids, rssis=ap_rssi_dict)
        zone = get_zone(zones=zones, co_ordinate=coord)
        print(str(zone))
        vector = distribution.get_vector(zone)
        vectors.append(vector)
        answers.append(zone)

    NormalizedMatrix.theAnswer = testing_data.answer  # JC-01 - used to pass the true answer around for run-time validation - used by dbg_combine_vector
    probability_zones = []
    resultant_vector = combine_vectors(answers, *vectors)
    measured_zone_1 = max(resultant_vector, key=resultant_vector.get)
    total = total_dict(resultant_vector)
    for item in resultant_vector.values():
        probability_zones.append(float(item / total))
    resultant_vector[measured_zone_1] = 0.0
    measured_zone_2 = max(resultant_vector, key=resultant_vector.get)

    return measured_zone_1, measured_zone_2, probability_zones


def show_position(img, x, y, color, probability=None):
    cv2.rectangle(img, (x - 50, y - 50), (x + 50, y + 50), (255, 0, 0), 2)
    if probability is not None:
        cv2.putText(img, probability, (x - 30, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.circle(img, (x, y), 5, color, thickness=-1)
    # rx = random.randrange(x - 60, x + 60, 1)
    # ry = random.randrange(y - 60, y + 60, 1)
    # cv2.circle(img, (rx, ry), 5, (255, 0, 0), thickness=-1)


def show_actual(img, zone, cood, color):
    cv2.putText(img, zone, cood, cv2.QT_FONT_NORMAL, 0.6, color, 1)


@app.route('/login', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('form-username')
        password = request.form.get('form-password')
        user = mongo.db.Users.find({'username': username, 'password': password}).count()
        if user == 0:
            response = "User Not Exist"
            print(response)
            return render_template('login.html', response=response)
        else:
            print(username + "found")
            train_data(dates=dates,
                       times=times,
                       num_combinations=num_combinations,
                       combination_modes=combination_modes,
                       error_modes=error_modes)
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
    global zone_data
    src = os.path.join(basePath, 'static/img', 'test.jpg')
    shutil.copyfile(os.path.join(basePath, 'static/img', 'test.jpg'), os.path.join(basePath, 'static/img', 'test1.jpg'))
    print("refresh")
    if zone_data:
        zone_data = []
    return redirect(url_for("update"))


@app.route("/wifi", methods=['GET', 'POST'])
def wifi():
    shutil.copyfile(os.path.join(basePath, 'static/img', 'test.jpg'), os.path.join(basePath, 'static/img', 'test1.jpg'))
    result_list = get_test_result(normalized_matrices, test_results)
    if request.method == 'POST':
        number = request.form.get('number')
        print(number)
        data = result_list[int(number)]
        BaseError = data['BaseError1']
        image = cv2.imread("static/img/test.jpg")
        i = 0
        for f in BaseError:
            show_position(image, 570 - i * 100, 300, (0, 255, 0), str(round(f, 2)))
            i += 1
        cv2.imwrite(os.path.join(basePath, 'static/img', 'test1.jpg'), image)
        data = {"AP": data['AP']}
        return json.dumps(data)

    # wifi = mongo.db.WifiScans.find()[0]
    # print(wifi['Scans'][0])
    # time = datetime.fromtimestamp(wifi['Time'] / 1e3).strftime("%m %d, %Y, %H:%M:%S")
    return render_template('wifi.html', data=result_list, img='./img/test1.jpg')


def train(sample_data):
    combination_method = get_combination_function(combination_modes[0])

    measured_zone = get_position(normalized=normalized_matrices[0],
                                 centroids=centroids,
                                 zones=zones,
                                 testing_data=sample_data,
                                 combination_method=combination_method)

    return measured_zone, sample_data.answer


def find_position(normalized: NormalizedMatrix,
                  centroids: List[Centroid],
                  zones: List[Zone],
                  data: Dict[AccessPoint, int],
                  combination_method: Callable) -> Tuple[Zone, Zone, List[float]]:
    combine_vectors = combination_method
    resultant = normalized.parent_matrix

    # JC-01 used the measured/reported zone instead of the actual zone to the algorithm.
    vectors = list()  # type: List[Dict[Zone, float]]
    answers = list()
    for distribution in resultant.normalizations:
        ap_rssi_dict = get_data_ap_combination(data, *distribution.access_points)
        print(str(ap_rssi_dict))
        coord = get_NNv4(centroid_points=centroids, rssis=ap_rssi_dict)
        zone = get_zone(zones=zones, co_ordinate=coord)
        print(str(zone))
        vector = distribution.get_vector(zone)
        vectors.append(vector)
        answers.append(zone)
    probability_zones = []
    resultant_vector = combine_vectors(answers, *vectors)
    measured_zone_1 = max(resultant_vector, key=resultant_vector.get)
    total = total_dict(resultant_vector)
    for item in resultant_vector.values():
        probability_zones.append(float(item / total))
    resultant_vector[measured_zone_1] = 0.0
    measured_zone_2 = max(resultant_vector, key=resultant_vector.get)

    return measured_zone_1, measured_zone_2, probability_zones


def get_record():
    zone_data

    return good_APs


def filter_scan_data(scan_data):
    samples = list()  # type: List[Sample]
    longest_rssis = 0
    bssid_rssi_dict = dict()  # type: Dict[AccessPoint, List[int]]
    for each_data in scan_data:
        for key, value in each_data.items():
            for access_point in access_points:
                if key == access_point.id:
                    bssid_rssi_dict[access_point] = value
                    if len(value) > longest_rssis:
                        longest_rssis = len(value)
    print(str(bssid_rssi_dict))
    print(str(longest_rssis))
    for index in range(longest_rssis):
        ap_rssi_dict = dict()  # type: Dict[AccessPoint, int]
        for key, rssis in bssid_rssi_dict.items():
            for access_point in access_points:
                if key == access_point:
                    try:
                        ap_rssi_dict[access_point] = rssis[index]
                    except IndexError:
                        # Hit because this AP may not have enough RSSI values. Append the average.
                        ap_rssi_dict[access_point] = round(sum(rssis) / len(rssis))
        samples.append(Sample(zones[5], ap_rssi_dict))

    return samples


def get_data_ap_combination(data: Dict[AccessPoint, int], *access_points: AccessPoint) -> Dict[AccessPoint, int]:
    return {k: v for k, v in data.items() if k in access_points}


def convert_epoch_to_datetime(epoch_time):
    return datetime.fromtimestamp(epoch_time / 1e3).strftime("%m %d, %Y, %H:%M:%S")


@app.route('/update', methods=['Get', 'POST', 'PUT'])
def update():
    global zone_data
    if request.method == 'PUT':
        global time
        # TODO: Perhaps store logs of what has occurred - wiping sensitive data?
        print(request.form.to_dict())
        dict_data = ast.literal_eval(request.form.getlist('PutData')[0])  # Gets the actual JSON data that was sent.
        print(dict_data)
        scan_data = dict_data["Scans"]["data"]
        scan_number = dict_data["Scans"]["Scan_number"]
        sample_data = filter_scan_data(scan_data)
        # sample_data1 = filter_access_points(dict_data["Scans"])
        scan_time = convert_epoch_to_datetime(dict_data["Time"])
        result = get_online_location(sample_data, scan_time)
        # new_time = dict_data["Time"]
        # print(new_time - time)
        # time = new_time
        # return str((200, "OK"))

        # response = get_position(dict_data)
        # print(str(response))
        # zone = response["closest_centroid"]
        # image = cv2.imread("static/img/test.jpg")
        # cv2.imwrite(os.path.join(basePath, 'static/img', 'test1.jpg'), image)
        return json.dumps(result)  # JSON-ifies the dictionary retrieved.

    if request.method == 'POST':
        number = request.form.get('number')
        data = location_data[int(number)]
        get_annotation(data, "Unknown")
        data = {"Zone": str(data[1])}
        return json.dumps(data)

    if zone_data:
        return render_template("location.html", zone_data=zone_data)
    else:
        return render_template("location.html")


@app.route('/Zone/<zone_num>')
def find_zone(zone_num):
    zone_num = int(zone_num) - 1
    data = create_test_data_list(access_points=access_points,
                                 zones=zones,
                                 folder_path=sorted_online_rssi_folder_path,
                                 dates=["November 23"],
                                 times=times)
    # sample_data = choice(data)
    zones_data = {}
    for zone in zones:
        zoneData = []
        for item in data:
            if item.answer == zone:
                zoneData.append(item)
        zones_data[zone] = zoneData
    print(str(zones_data))
    executor.submit(get_location, zones_data, zones[zone_num])

    return render_template("location.html")


def get_annotation(result, actual_zone):
    zone1 = result[0]
    zone2 = result[1]
    probabilities = result[2]
    print(str(probabilities))
    image = cv2.imread("static/img/test.jpg")
    image1 = cv2.imread("static/img/test.jpg")
    for i in range(5):
        show_position(image, 570 - (100 * i), 300, (0, 0, 255), str(round(probabilities[i] * 100, 2)) + "%")
        if zone1 == zones[i]:
            show_position(image1, 570 - (100 * i), 300, (0, 255, 0))
            show_position(image, 570 - (100 * i), 300, (0, 255, 0))
        if zone2 == zones[i]:
            show_position(image1, 570 - (100 * i), 300, (0, 255, 255))
            show_position(image, 570 - (100 * i), 300, (0, 255, 255))
        if actual_zone == zones[i]:
            show_actual(image, str(actual_zone), (540 - (i * 100), 370), (0, 255, 0))
            show_actual(image1, str(actual_zone), (540 - (i * 100), 370), (0, 255, 0))

    cv2.imwrite(os.path.join(basePath, 'static/img', 'test1.jpg'), image)
    cv2.imwrite(os.path.join(basePath, 'static/img', 'test2.jpg'), image1)
    print("Images changed")


def get_online_location(sample_data, scan_time):
    print("start locating")
    combination_method = get_combination_function(combination_modes[0])
    for item in sample_data:
        time.sleep(2)
        result = find_position(normalized_matrices[0], centroids, zones, item.scan, combination_method)
        broadcaster_data = {'Actual_Zone': "Unknown",
                            'Primary_Zone': str(result[0]),
                            'Secondary_Zone': str(result[1]),
                            'Time': scan_time
                            }
        zone_data.append(broadcaster_data)
        location_data.append(result)
        get_annotation(result, "Unknown")
        push_once(broadcaster_data)
    return {"answer": "Zone 1"}


def get_location(data, zone):
    zone_counter = {"Zone 1": 0,
                    "Zone 2": 0,
                    "Zone 3": 0,
                    "Zone 4": 0,
                    "Zone 5": 0}
    zone_counter1 = {"Zone 1": 0,
                     "Zone 2": 0,
                     "Zone 3": 0,
                     "Zone 4": 0,
                     "Zone 5": 0}
    print(str(zone) + " started")
    for key, value in data.items():
        if key is not zone:
            continue
        for i in range(35):
            time.sleep(2)
            secondsSinceEpoch = time.time()
            timeObj = time.localtime(secondsSinceEpoch)
            timeNow = '%d-%d-%d %d:%d:%d' % (
                timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
            results = train(choice(value))
            print(str(results))
            zone_measured = results[0]
            zone_actual = results[1]
            zone_counter[str(zone_measured[0])] += 1
            zone_counter1[str(zone_measured[1])] += 1
            broadcaster_data = {'Actual_Zone': str(zone_actual),
                                'Primary_Zone': str(zone_measured[0]),
                                'Secondary_Zone': str(zone_measured[1]),
                                'Time': timeNow
                                }
            print(broadcaster_data)
            zone_data.append(broadcaster_data)
            location_data.append(results)
            get_annotation(results[0], str(zone_actual))
            push_once(broadcaster_data)
            print(zone_counter)
            print(zone_counter1)
    print("done")


@app.route('/push')
def push_once(broadcaster_data):
    event_name = 'message'
    print("publish msg==>", broadcaster_data)
    socketio.emit(event_name, broadcaster_data, broadcast=True, namespace=name_space)
    return 'send msg successful!'


@socketio.on('recevice message', namespace=name_space)
def test_message(message):
    print('recevice message', message)
    # emit('message', {'data': message['data']})


@socketio.on('connect', namespace=name_space)
def connected_msg():
    """客户端连接"""
    print('client connected!', request.sid)
    socketio.emit('abcde', 'hello', namespace=name_space)


@socketio.on('disconnect', namespace=name_space)
def disconnect_msg():
    """客户端离开"""
    print('client disconnected!')


# region Main
if __name__ == '__main__':
    app.debug = True
    app.run()
# endregion
