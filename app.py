import ast
import json
import operator
import os
import shutil
import simpleaudio as sa
from collections import OrderedDict
from datetime import timedelta, datetime
from math import floor
from os import makedirs, listdir
from os.path import isdir, isfile, join
from random import choice, random, sample, seed, shuffle
from statistics import mean
from urllib import request
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy
import numpy as np
import pandas as pd
import xlrd
from flask import Flask, send_from_directory, session
from flask_pymongo import PyMongo
from flask import Flask, render_template, flash, redirect, url_for, make_response
from flask import request
from celery import Celery

from Algorithms.NearestNeighbour.Calculation import get_calculation_function
from Algorithms.NearestNeighbour.KNNv1 import get_KNNv1
from Algorithms.NearestNeighbour.NNv4 import get_NNv4_RSSI
from Algorithms.svm.svm import svm_model
from Algorithms.svm.svmutil import svm_predict
from Resources.Objects.Building import Building
from Resources.Objects.Comparesheet import Comparesheet
from Resources.Objects.Floor import Floor
from Resources.Objects.Matrices.CombinedDistribution import test_combination_matrices, test_svm_matrices, \
    test_normalized_dict
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix, sort_matrices
from Algorithms.Combination.Combination import get_combination_function, build_combined_distributions
from Resources.Objects.Matrices.ProbabilityDistribution import ProbabilityMatrix
from Resources.Objects.Points.AccessPoint import AccessPoint, get_n_ap_combinations, get_ap_combinations
from Resources.Objects.Points.GridPoint import GridPoint
from Resources.Objects.Points.Centroid import Centroid
from Offline.MatrixProduction import create_all_matrices_from_rssi_data, get_NNv4
from Resources.Objects.Room import Room
from Resources.Objects.TestData import create_test_data_list, TestResult, Sample
from Resources.Objects.Zone import *
from Resources.Objects.Points.Point import Point
from flask_socketio import SocketIO, emit
from Resources.Objects.Worksheet import Worksheet
from Algorithms.Combination.AdaptiveBoosting import create_matrices
from typing import List, Tuple, Dict, Callable
from time import time
from uuid import uuid4
import xlsxwriter.exceptions
import xlsxwriter
import KFold
import math
import libsvm

ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}
executor = ThreadPoolExecutor(2)
basePath = os.path.dirname(__file__)
position_data = list()


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
app.send_file_max_agedefault = timedelta(seconds=1)
x = 0
y = 0

# Globals to set parameters for Matrix Production
dates = {
    "HOME": ["April 3", "April 6", "April 8"],
    "SCAET": ["November 19", "November 20", "November 21"]
}
times = ["15_00", "18_00", "20_00"]  # readonly
num_combinations = [2]
combination_modes = ["WGT"]  # ["AVG", "WGT", "AB"]
error_modes = ["MAX"]
locate_modes = ["NNv4", "kNN"]

# 1. Establish file location data.
main_folder = "./Data/November"
access_point_file_path = "{}/Points/Access Points/Access Points.csv".format(main_folder)
grid_point_file_path = "{}/Points/Grid Points/November 19 - November 20 - November 21 - November 23 Grid Points.csv".format(
    main_folder)
centroid_file_path = "{}/Points/Centroid Points/Centroid Points.csv".format(main_folder)
zone_file_path = "{}/Zones/Zones.csv".format(main_folder)
sorted_offline_rssi_folder_path = "{}/RSSI Data/Test Data/Offline".format(main_folder)
sorted_online_rssi_folder_path = "{}/RSSI Data/Test Data/Online".format(main_folder)

# 2. Instantiate "static" objects.
# access_points = AccessPoint.create_point_list(file_path=access_point_file_path)
# grid_points = GridPoint.create_point_list(file_path=grid_point_file_path, access_points=access_points)
# centroids = Centroid.create_point_list(file_path=centroid_file_path, grid_points=grid_points)
# zones = get_all_zones(file_path=zone_file_path)
# APs = list(mongo.db.Access_Points.find())
# GPs = list(mongo.db.Grid_Points.find())
# CDs = list(mongo.db.Centroid.find())
# ZOs = list(mongo.db.Zones.find())
# access_points = AccessPoint.create_db_point_list(APs)
# grid_points = GridPoint.create_point_list_db(GPs, access_points)
# centroids = Centroid.create_point_list_db(CDs, grid_points)
# zones = get_all_zones_db(ZOs)
data = list(mongo.db.Data.find())
start_building_time = time()
buildings = Building.create_building_list(data)
end_building_time = time()
print("Building was created: {}s.".format(end_building_time - start_building_time))
current_user = dict()
# normalized_matrices = List[NormalizedMatrix]
zone_data = []
position_data = []
location_data = []
room_list = []


# training_data = list()  # type: List[Sample]
# testing_data = list()  # type: List[Sample]

def define_data(data: List[Sample], aps: List[AccessPoint]):
    floor_ap = dict()
    for ap in aps:
        min = 0
        max = 0
        data_list = list()
        for d in data:
            if ap in d.scan:
                if data.index(d) == 0:
                    min = d.scan[ap]
                    max = d.scan[ap]
                if d.scan[ap] > max:
                    max = d.scan[ap]
                if d.scan[ap] < min:
                    min = d.scan[ap]
                data_list.append(d.scan[ap])
        avg = floor(mean(data_list))
        floor_ap[ap] = (min, avg, max)
    return floor_ap


def random_data(floor: Floor, online_pct: float, samples: List[Sample]):
    online_samples = list()
    offline_samples = list()
    for zone in floor.zones:
        new_samples = list()
        for sample in samples:
            if sample.answer == zone:
                new_samples.append(sample)
        sample_num = int(len(new_samples) * online_pct)
        shuffle(new_samples)  # Shuffle the data.
        online_part = new_samples[:sample_num]
        offline_part = new_samples[sample_num:]
        online_samples += online_part
        offline_samples += offline_part
    return online_samples, offline_samples


def mse_calculation(actual_zones, predict_zones):
    summation = 0  # variable to store the summation of differences
    n = len(actual_zones)  # finding total number of items in list
    for i in range(0, n):  # looping through each element of the list
        actual_cood = actual_zones[i].center
        predict_cood = predict_zones[i].center
        squared_difference = (actual_cood[0] - predict_cood[0]) ** 2 + (
                actual_cood[1] - predict_cood[1]) ** 2  # taking square of the differene
        summation += math.sqrt(squared_difference)  # taking a sum of all the differences
    MSE = summation / n  # dividing summation by total values to obtain average
    # print("The Mean Square Error is: ", MSE)
    return MSE


def compare_methods(buildings: buildings,
                    num_combination: int,
                    combination_mode: str,
                    location_modes: [str],
                    testing_modes: [str],
                    error_mode: str,
                    write_sheet: bool,
                    num_splits: int
                    ):
    compare_sheets = list()  # type: List[Comparesheet]
    work_sheets = list()  # type: List[Worksheet]
    data_results = dict()
    for building in buildings:
        if building.building_name == "SCAET":
            continue
        for floor in building.floors:
            # if floor.floor_id == "5":
            #     continue
            centroids = floor.get_centroids
            grid_points = floor.grid_points
            full_ap_list = floor.access_points
            raw_data_1 = floor.data
            data_result = define_data(raw_data_1, full_ap_list)
            data_results[floor.floor_id] = data_result
            zones = floor.zones
            type_mode = ""
            # Set mode:

            print("get new Train data")
            divide_data = random_data(floor, 0.4, raw_data_1)
            testing_data = divide_data[0]
            training_data = divide_data[1]
            shuffle(training_data)
            shuffle(testing_data)
            sheet_tab = dict()
            print("Reserving {} scans for final testing.".format(len(testing_data)))
            ap_type = 0
            access_point_d = []
            combine = False

            while ap_type < 3:
                if ap_type == 0:
                    combine = False
                    type_mode = "AP"
                    access_points = [access_point for access_point in floor.access_points if
                                     access_point.type == type_mode]
                elif ap_type == 1:
                    if floor.floor_id == "1":
                        break
                    else:
                        combine = False
                        type_mode = "Beacon"
                        access_points = [access_point for access_point in floor.access_points if
                                         access_point.type == type_mode]
                else:
                    type_mode = "Mix"
                    access_points = floor.access_points
                    combine = False

                NormalizedMatrix.error_mode = error_mode
                NormalizedMatrix.combination_mode = combination_mode
                NormalizedMatrix.type_mode = type_mode

                start_mm_time = time()
                max_mean = dict()
                best_ap = dict()
                total_ap = dict()
                for zone in zones:
                    max_zone = dict()
                    zone_scan = [sample.scan for sample in training_data if sample.answer == zone]
                    for ap in access_points:
                        value_list = [scan[ap] for scan in zone_scan if ap in scan]
                        mean_value = mean(value_list)
                        max_zone[ap] = mean_value
                    sorted_max_zone = sorted(max_zone.items(), key=lambda kv: kv[1], reverse=True)
                    sorted_max_zone = dict((key, value) for key, value in sorted_max_zone)
                    max_mean[zone] = sorted_max_zone
                for ap in access_points:
                    ap_list = [rssi[ap] for rssi in max_mean.values()]
                    mean_ap = mean(ap_list)
                    total_ap[ap] = mean_ap
                sorted_total_ap = sorted(total_ap.items(), key=lambda kv: kv[1], reverse=True)
                sorted_total_ap = [key for key, value in sorted_total_ap]
                for d in range(3, len(access_points) + 1):
                    sorted_d_ap = sorted_total_ap[:d]
                    best_ap[d] = sorted(sorted_d_ap)
                end_mm_time = time()
                mm_training_time = end_mm_time - start_mm_time
                print("The Access Points use: {}".format(access_points))
                access_point_d.append(len(access_points))
                gd_table_list = dict()
                jc_table_list = dict()
                mm_table_list = dict()
                rd_table_list = dict()
                ig_table_list = dict()
                for location_mode in location_modes:
                    testing_modes = ["SVM", "NNv4"]
                    NormalizedMatrix.train_mode = location_mode
                    print("starting training with {}".format(location_mode))
                    NormalizedMatrix.location_mode = location_mode
                    test_results = KFold.create_compare_combination(num_splits=num_splits,
                                                                    mm_ap_dict=best_ap,
                                                                    combination_mode=combination_mode,
                                                                    location_mode=location_mode,
                                                                    num_combinations=num_combination,
                                                                    training_data=training_data,
                                                                    access_points=access_points,
                                                                    zones=zones,
                                                                    grid_points=grid_points,
                                                                    centroids=centroids,
                                                                    combine=combine)
                    best_jc_model = test_results[0]
                    best_gd_model = test_results[1]
                    gd_training_time = test_results[3]
                    jc_training_time = test_results[2]
                    best_ig_list = test_results[5]
                    svm_models = test_results[4]
                    ig_training_time = test_results[6]
                    normalized_dict = test_results[7]

                    jc_test_results = test_normalized_dict(normalized_dict=normalized_dict,
                                                           centroids=centroids,
                                                           zones=zones,
                                                           testing_data=testing_data)

                    if ap_type == 2:
                        # skip Beacon
                        work_sheets.append(Worksheet(num_combinations=num_splits,
                                                     error_mode="Max",
                                                     building=building,
                                                     floor=floor,
                                                     ap_type=type_mode,
                                                     combination_mode=combination_mode,
                                                     location_mode=location_mode,
                                                     normalized_probability_matrices=normalized_dict,
                                                     test_results=jc_test_results))
                    test_features = list()  # type: List[Dict[AccessPoint, int]]
                    test_classes = list()  # type: List[int]
                    actual_zone = list()
                    for sample in testing_data:
                        test_features.append(sample.scan)
                        actual_zone.append(sample.answer)
                        test_classes.append(sample.answer.num)
                    combination_method = get_combination_function(combination_mode)
                    if location_mode == "kNNv1" or location_mode == "kNNv2" or location_mode == "kNNv3":
                        testing_modes = [location_mode]
                    for testing_mode in testing_modes:
                        gd_correct = list()
                        jc_correct = list()
                        mm_correct = list()
                        rd_correct = list()
                        ig_correct = list()

                        gd_testing_time = list()
                        jc_testing_time = list()
                        random_ap_tuple = dict()
                        # NormalizedMatrix.test_mode = testing_mode
                        for d in range(3, len(access_points) + 1):
                            access_point_combinations = get_n_ap_combinations(access_points,
                                                                              d)  # type: List[Tuple[AccessPoint, ...]]
                            random_ap_tuple[d] = choice(access_point_combinations)
                        # print("starting testing with {}".format(testing_mode))
                        start_mm_test_time = time()
                        for d, ap_list in best_ap.items():
                            ap_tuple = tuple(sorted(ap_list))
                            correct = 0
                            mm_predict = list()
                            for sample in testing_data:
                                filter_sample = dict(
                                    (ap, rssi) for ap, rssi in sample.scan.items() if ap in access_points)
                                # sorted_data = sorted(filter_sample.items(), key=lambda kv: kv[1], reverse=True)
                                # selected_ap = [ap for ap, rssi in sorted_data[:d]]
                                # d_data = dict((ap, rssi) for ap, rssi in filter_sample.items() if ap in selected_ap)
                                d_data = get_data_ap_combination(filter_sample, *ap_tuple)
                                d_svm = svm_models[ap_tuple]
                                calculated_zone = other_test(floor.get_centroids, testing_mode, floor.zones,
                                                             floor.grid_points, d_data, d_svm)
                                mm_predict.append(calculated_zone)
                                if calculated_zone == sample.answer:
                                    correct += 1
                            mm_mse = mse_calculation(actual_zones=actual_zone, predict_zones=mm_predict)
                            mm_accuracy = correct / len(testing_data)
                            mm_correct.append({ap_tuple: (mm_accuracy, mm_mse)})
                        end_mm_test_time = time()
                        mm_testing_time = end_mm_test_time - start_mm_test_time
                        for d, ap_tuple in random_ap_tuple.items():
                            correct = 0
                            for sample in testing_data:
                                filter_sample = dict(
                                    (ap, rssi) for ap, rssi in sample.scan.items() if ap in access_points)
                                d_data = get_data_ap_combination(filter_sample, *ap_tuple)
                                d_svm = svm_models[ap_tuple]
                                calculated_zone = other_test(floor.get_centroids, testing_mode, floor.zones,
                                                             floor.grid_points, d_data, d_svm)
                                if calculated_zone == sample.answer:
                                    correct += 1
                            rd_accuracy = correct / len(testing_data)
                            rd_correct.append({ap_tuple: rd_accuracy})
                        start_ig_test_time = time()
                        for d in range(3, len(access_points) + 1):
                            ig_tuple = tuple(sorted(best_ig_list[:d]))
                            correct = 0
                            ig_predict = list()
                            for sample in testing_data:
                                filter_sample = dict(
                                    (ap, rssi) for ap, rssi in sample.scan.items() if ap in access_points)
                                d_data = get_data_ap_combination(filter_sample, *ig_tuple)
                                d_svm = svm_models[ig_tuple]
                                calculated_zone = other_test(floor.get_centroids, testing_mode, floor.zones,
                                                             floor.grid_points, d_data, d_svm)
                                ig_predict.append(calculated_zone)
                                if calculated_zone == sample.answer:
                                    correct += 1
                            ig_mse = mse_calculation(actual_zones=actual_zone, predict_zones=ig_predict)
                            ig_accuracy = correct / len(testing_data)
                            ig_correct.append({ig_tuple: (ig_accuracy, ig_mse)})
                        end_ig_test_time = time()
                        ig_testing_time = end_ig_test_time - start_ig_test_time
                        if testing_mode == "SVM":
                            for ap_tuple, model in best_jc_model.items():
                                start_jc_testing_time = time()
                                svm_dict = model[1]
                                test_predict = list()
                                correct = 0
                                jc_predict = list()
                                for ap, svm_list in svm_dict.items():
                                    aps_being_used = [x for x in ap]
                                    ap_test_features = list()
                                    for feature_set in test_features:
                                        ap_test_features.append(
                                            [value for key, value in feature_set.items() if key in aps_being_used])
                                    for svm in svm_list:
                                        p_labs, p_acc, p_vals = svm_predict(y=test_classes, x=ap_test_features,
                                                                            m=svm,
                                                                            options="-q")
                                        test_predict.append(p_labs)
                                        # print(p_acc[0])
                                        # average_accuracy.append(p_acc[0])
                                for d in range(len(test_predict[0])):
                                    zone_predictions = list()
                                    for p in test_predict:
                                        zone_predictions.append(p[d])
                                    best_predict_zone = max(zone_predictions, key=zone_predictions.count)
                                    predicted_zone = zones[int(best_predict_zone) - 1]
                                    jc_predict.append(predicted_zone)
                                    if predicted_zone.num == test_classes[d]:
                                        correct += 1
                                jc_mse = mse_calculation(actual_zones=actual_zone, predict_zones=jc_predict)
                                jc_accuracy = correct / len(testing_data)
                                # highest_accuracy = max(average_accuracy)
                                # jc_correct[ap_tuple] = accuracy
                                jc_correct.append({ap_tuple: (jc_accuracy, jc_mse)})
                                end_jc_testing_time = time()
                                jc_testing_time.append(end_jc_testing_time - start_jc_testing_time)
                            for ap, model in best_gd_model.items():
                                start_gd_testing_time = time()
                                svm_list = model[1]
                                test_predict = list()
                                correct = 0
                                aps_being_used = [x for x in ap]
                                ap_test_features = list()
                                gd_predict = list()
                                for feature_set in test_features:
                                    ap_test_features.append(
                                        [value for key, value in feature_set.items() if key in aps_being_used])
                                for svm in svm_list:
                                    p_labs, p_acc, p_vals = svm_predict(y=test_classes, x=ap_test_features, m=svm,
                                                                        options="-q")
                                    test_predict.append(p_labs)
                                for d in range(len(test_predict[0])):
                                    zone_predictions = list()
                                    for p in test_predict:
                                        zone_predictions.append(p[d])
                                    best_predict_zone = max(zone_predictions, key=zone_predictions.count)
                                    predicted_zone = zones[int(best_predict_zone) - 1]
                                    gd_predict.append(predicted_zone)
                                    if predicted_zone.num == test_classes[d]:
                                        correct += 1
                                gd_mse = mse_calculation(actual_zones=actual_zone, predict_zones=gd_predict)
                                gd_accuracy = correct / len(testing_data)
                                # gd_correct[ap] = accuracy
                                gd_correct.append({ap: (gd_accuracy, gd_mse)})
                                end_gd_testing_time = time()
                                gd_testing_time.append(end_gd_testing_time - start_gd_testing_time)
                            # for i in range(2, len(access_points)+1):
                            #     ap_list = [ap for ap in sorted_best_ap.keys()]
                            #     ap_selected = ap_list[:i]
                            #     ap_test_features = list()
                            #     for feature_set in test_features:
                            #         ap_test_features.append(
                            #             [value for key, value in feature_set.items() if key in ap_selected])

                            print(
                                "JC testing time is {}, GD testing time is {}".format(sum(jc_testing_time),
                                                                                      sum(gd_testing_time)))
                        else:
                            for ap_tuple, model in best_jc_model.items():
                                start_jc_testing_time = time()
                                correct = 0
                                normalized_matrix = model[0]
                                jc_predict = list()
                                # if location_mode != testing_mode:
                                #     if len(ap_tuple) > len(access_points):
                                #         break
                                for sample in testing_data:
                                    zone_list = find_position(normalized_matrix, floor.get_centroids,
                                                              floor.zones, testing_mode, floor.grid_points,
                                                              sample.scan, combination_method)
                                    jc_predict.append(zone_list)
                                    if zone_list == sample.answer:
                                        correct += 1
                                jc_mse = mse_calculation(actual_zones=actual_zone, predict_zones=jc_predict)
                                jc_accuracy = correct / len(testing_data)
                                jc_correct.append({ap_tuple: (jc_accuracy, jc_mse)})
                                end_jc_testing_time = time()
                                jc_testing_time.append(end_jc_testing_time - start_jc_testing_time)
                            for ap_tuple, model in best_gd_model.items():
                                start_gd_testing_time = time()
                                normalized_matrix = model[0]
                                correct = 0
                                gd_predict = list()
                                for sample in testing_data:
                                    zone_list = find_position(normalized_matrix, floor.get_centroids,
                                                              floor.zones, testing_mode, floor.grid_points,
                                                              sample.scan, combination_method)
                                    gd_predict.append(zone_list)
                                    if zone_list == sample.answer:
                                        correct += 1
                                gd_mse = mse_calculation(actual_zones=actual_zone, predict_zones=gd_predict)
                                gd_accuracy = correct / len(testing_data)
                                gd_correct.append({ap_tuple: (gd_accuracy, gd_mse)})
                                end_gd_testing_time = time()
                                gd_testing_time.append(end_gd_testing_time - start_gd_testing_time)
                            print(
                                "JC testing time is {}, GD testing time is {}".format(sum(jc_testing_time),
                                                                                      sum(gd_testing_time)))
                        print("Finished testing with {}".format(testing_mode))
                        gd_table_list[(location_mode, testing_mode)] = gd_correct, gd_testing_time, gd_training_time
                        jc_table_list[(location_mode, testing_mode)] = jc_correct, jc_testing_time, jc_training_time
                        mm_table_list[(location_mode, testing_mode)] = mm_correct, [mm_testing_time], mm_training_time
                        rd_table_list[(location_mode, testing_mode)] = rd_correct, [0.0], 0.0
                        ig_table_list[(location_mode, testing_mode)] = ig_correct, [ig_testing_time], ig_training_time
                sheet_tab[ap_type] = [gd_table_list, jc_table_list, mm_table_list, ig_table_list]
                ap_type += 1

            compare_sheets.append(Comparesheet(num_combinations=num_combination,
                                               error_mode=error_mode,
                                               best_gamma="Default",
                                               train_data=len(training_data),
                                               test_data=len(testing_data),
                                               k_fold=num_splits,
                                               building=building,
                                               floor=floor,
                                               tables=sheet_tab,
                                               access_points=access_point_d,
                                               type_mode=type_mode
                                               ))

    if write_sheet:
        # Reset the point objects:
        Point.reset_points()
        excel_start_time = time()

        file_name = "Compare-ALL-details-chart-820.xlsx"
        excel_workbook = xlsxwriter.Workbook("{}/Matrices/{}".format(main_folder, file_name))
        bold = excel_workbook.add_format({'bold': True})
        merge_format = excel_workbook.add_format({'bold': True, 'align': 'center'})
        high_light = excel_workbook.add_format({'bold': True, 'align': 'center', 'bg_color': 'yellow'})

        # Save the key page:
        excel_worksheet = excel_workbook.add_worksheet("Keys")
        # Worksheet.save_key_page(excel_worksheet, bold=bold, merge_format=merge_format)
        Comparesheet.save_key_page(excel_worksheet, data_results, bold=bold, merge_format=merge_format)

        # Save all other pages:
        problems_saving = list()  # type: List[str]
        for sheet in compare_sheets:
            try:
                excel_worksheet = excel_workbook.add_worksheet(sheet.tab_title)
                chart_worksheet = excel_workbook.add_worksheet("{}-chart".format(sheet.tab_title))
                special_worksheet = excel_workbook.add_worksheet("{}-special".format(sheet.tab_title))
                # matrix_worksheet = excel_workbook.add_worksheet("{}-matrix".format(sheet.tab_title))
            except xlsxwriter.exceptions.DuplicateWorksheetName:

                problem_sheet = sheet.tab_title
                problem_resolution = str(uuid4())[:25]
                problem_description = "Worksheet {} has the same tab-title as another page. It has been replaced with {}."
                problems_saving.append(problem_description.format(problem_sheet, problem_resolution))

                excel_worksheet = excel_workbook.add_worksheet(problem_resolution)

            sheet.save(excel_worksheet, chart_worksheet, special_worksheet, excel_workbook, bold=bold,
                       merge_format=merge_format)

        for sheet in work_sheets:
            try:
                excel_worksheet = excel_workbook.add_worksheet("{}-matrix".format(sheet.tab_title))
            except xlsxwriter.exceptions.DuplicateWorksheetName:

                problem_sheet = sheet.tab_title
                problem_resolution = str(uuid4())[:25]
                problem_description = "Worksheet {} has the same tab-title as another page. It has been replaced with {}."
                problems_saving.append(problem_description.format(problem_sheet, problem_resolution))

                excel_worksheet = excel_workbook.add_worksheet(problem_resolution)

            sheet.save(excel_worksheet, bold=bold,
                       merge_format=merge_format, high_light=high_light)

        excel_workbook.close()
        excel_end_time = time()

        print("Workbook Write Time: {}s.".format(excel_end_time - excel_start_time))
        # print("Sample time: {}s".format(train_time))
        if len(problems_saving) > 0:
            print("There were problems saving {} worksheets.".format(len(problems_saving)))
            for problem in problems_saving:
                print(problem)
            print("You can manually change the tab-titles now if desired.")


def sort_test_result(test_results: Dict[NormalizedMatrix, TestResult]):
    new_test = OrderedDict()  # type: OrderedDict[NormalizedMatrix, TestResult]
    results = list(test_results.values())
    results.sort(key=lambda result: result, reverse=True)
    reversed_test = {v: k for k, v in test_results.items()}
    for result in results:
        distribution = reversed_test[result]
        new_test[distribution] = result
    return new_test


def check_and_create_folder(folder_path: str):
    if isdir(folder_path):
        return
    makedirs(folder_path)


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


@app.route('/logout', methods=['Get'])
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def login():
    global current_user
    if request.method == 'POST':
        username = request.form.get('form-username')
        password = request.form.get('form-password')
        user = mongo.db.Users.find({'username': username, 'password': password})
        if user.count() == 0:
            response = "User Not Exist"
            print(response)
            return render_template('login.html', response=response)
        else:
            current_user = user[0]
            session['uname'] = username
            resp = redirect(url_for('home'))
            if 'isSaved' in request.form:
                resp.set_cookie('uname', username, 60 * 60 * 24 * 365)
            print(username + "found")
            return resp
    if request.method == 'GET':
        if 'uname' in session:
            uname = session['uname']
            return redirect(url_for('home'))
        else:
            if 'uname' in request.cookies:
                uname = request.cookies.get('uname')
                session['uname'] = uname
                return redirect(url_for('home'))
            else:
                return render_template('login.html')


@app.route("/setting", methods=["Get", "POST"])
def setting():
    if request.method == 'POST':
        selection_mode = request.form.get('selection_mode')
        type_mode = request.form.get('type_mode')
        train_mode = request.form.get('train_mode')
        test_mode = request.form.get('test_mode')
        num_combination = int(request.form.get("num_combination"))
        num_splits = int(request.form.get("num_splits"))
        train_db(train_mode=train_mode,
                 test_mode=test_mode,
                 selection_mode=selection_mode,
                 num_combination=num_combination,
                 num_splits=num_splits,
                 type_mode=type_mode)
        return {"data": "data Changed"}
    return render_template("setting.html", building_list=buildings,
                           combination_mode=NormalizedMatrix.combination_mode,
                           test_mode=NormalizedMatrix.test_mode, train_mode=NormalizedMatrix.train_mode,
                           type_mode=NormalizedMatrix.type_mode)


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    directory = "C:/Users/tongche/Desktop/New Project/SheridanIPS_Server/Data/November/Matrices"
    return send_from_directory(directory=directory, filename=filename)


@app.route("/details/<room_num>", methods=["Get", "POST"])
def details(room_num):
    room = room_list[int(room_num)]
    return render_template("Details.html", room=room)


@app.route("/home", methods=["Get", "Post"])
def home():
    if request.method == "POST":
        compare_methods(buildings=buildings,
                        num_combination=4,
                        combination_mode="WGT",
                        location_modes=["SVM", "NNv4"],
                        error_mode=error_modes[0],
                        write_sheet=True,
                        num_splits=5,
                        testing_modes=["SVM", "NNv4"]
                        )
    user_type = current_user['role']
    if user_type == 'admin':
        return render_template("home_admin.html")
    else:
        return render_template("home.html")


def find_building(scan_data):
    ap_list = list()
    for each_data in scan_data:
        ap_list.append(each_data['BSSID'])
    match_rate = dict()
    for building in buildings:
        counter = 0
        for ap in ap_list:
            if ap in building.access_points:
                counter += 1
        match_rate[building] = counter
    target_building = max(match_rate, key=match_rate.get)
    return target_building


def find_floor(filtered_data, target_building):
    ap_filtered = list()
    for each_data in filtered_data:
        ap_filtered.append(each_data['BSSID'])
    floor_list = list()
    floors = target_building.floors
    for f in floors:
        match_rate = {"Floor": f}
        counter = 0
        for ap in ap_filtered:
            if ap in f.ap_list:
                counter += 1
        match_rate["Count"] = counter
        floor_list.append(match_rate)
    for floor in floor_list:
        print(floor["Floor"].floor_id + ": " + str(floor["Count"]))
    target_floor = max(floor_list, key=lambda item: item["Count"])
    return target_floor["Floor"]


def sound(x, z):
    frequency = x  # Our played note will be 440 Hz
    fs = 44100  # 44100 samples per second
    seconds = z  # Note duration of 3 seconds

    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, seconds * fs, False)

    # Generate a 440 Hz sine wave
    note = np.sin(frequency * t * 2 * np.pi)

    # Ensure that highest value is in 16-bit range
    audio = note * (2 ** 15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, fs)

    # Wait for playback to finish before exiting
    play_obj.wait_done()


@app.route("/map", methods=["Get", "POST", "PUT"])
def show_map():
    global position_data
    if request.method == 'PUT':
        frequency = 4000
        duration = 1000
        # winsound.Beep(frequency, duration)
        sound(300, 2)
        dict_data = ast.literal_eval(request.form.getlist('PutData')[0])  # Gets the actual JSON data that was sent.
        user_id = dict_data['ID']
        scan_data = dict_data['Scans']['data']
        # scan_number = dict_data["Scans"]["Scan_number"]
        target_building = find_building(scan_data)
        print("The Target building is " + target_building.building_name)
        filtered_data = [d for d in scan_data if int(mean(d['RSSIs'])) > -85]
        target_floor = find_floor(filtered_data, target_building)
        print("The Target building is " + target_floor.floor_id)
        samples = filter_scan_data(scan_data, target_floor.access_points, target_floor.zones)
        combination_method = get_combination_function(combination_modes[0])
        results = list()
        for sample in samples:
            zone_list = find_position(combination_method=combination_method, centroids=target_floor.get_centroids,
                                      data=sample.scan, normalized=target_floor.matrix,
                                      grid_points=target_floor.grid_points,
                                      location_mode="NNv4", zones=target_floor.zones)
            results.append(zone_list)
        for result in results:
            print(result)
        target_zone = max(results, key=results.count)
        print("The Target Zone is " + target_zone.zone_num)
        target_room = target_zone.room_id
        user_data = {
            'Building': str(target_building.building_name),
            'Floor': target_floor.floor_id,
            'Room': target_room,
            'Zone': str(target_zone)
        }
        mongo.db.Users.find_one_and_update({'id': user_id}, {'$set': user_data})
        target_user = mongo.db.Users.find({'id': user_id})[0]
        user_list = list(mongo.db.Users.find())
        position_data = user_list
        target_user.pop('_id')
        push_once(target_user)
        return target_user
    if request.method == 'POST':
        return
    user_type = current_user['role']
    if user_type == 'admin':
        return render_template("map_admin.html", position_data=position_data)
    else:
        return render_template("map.html", position_data=position_data)


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


@app.route("/location", methods=["Get", "Post"])
def location():
    global zone_data
    src = os.path.join(basePath, 'static/img', 'test.jpg')
    shutil.copyfile(os.path.join(basePath, 'static/img', 'test.jpg'),
                    os.path.join(basePath, 'static/img', 'test1.jpg'))
    print("refresh")
    if zone_data:
        zone_data = []
    return redirect(url_for("update"))


@app.route("/wifi", methods=['GET', 'POST'])
def wifi():
    shutil.copyfile(os.path.join(basePath, 'static/img', 'test.jpg'),
                    os.path.join(basePath, 'static/img', 'test1.jpg'))
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
    return render_template('task.html', data=result_list, img='./img/test1.jpg')


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
                  location_mode: str,
                  grid_points: List[GridPoint],
                  data: Dict[AccessPoint, int],
                  combination_method: Callable):
    if "U" in normalized.id:
        combine_vectors = combination_method
        resultant = normalized.parent_matrix
        # JC-01 used the measured/reported zone instead of the actual zone to the algorithm.
        vectors = list()  # type: List[Dict[Zone, float]]
        answers = list()

        for distribution in resultant.normalizations:
            ap_rssi_dict = get_data_ap_combination(data, *distribution.access_points)
            location_method = get_calculation_function(location_mode)
            if location_mode == "NNv4" or location_mode == "kNNv2" or location_mode == "kNNv1":
                calculated_co_ordinate = location_method(centroid_points=centroids, rssis=ap_rssi_dict)
            if location_mode == "kNNv3":
                calculated_co_ordinate = location_method(grid_points=grid_points, rssis=ap_rssi_dict)
            # coord = get_NNv4_RSSI(centroid_points=centroids, rssis=ap_rssi_dict)
            zone = get_zone(zones=zones, co_ordinate=calculated_co_ordinate)
            # print(str(zone))
            vector = distribution.get_vector(zone)
            vectors.append(vector)
            answers.append(zone)
        probability_zones = []
        resultant_vector = combine_vectors(answers, *vectors)
        measured_zone_1 = max(resultant_vector, key=resultant_vector.get)
        # total = total_dict(resultant_vector)
        # print(resultant_vector)
        # for item in resultant_vector.values():
        #     probability_zones.append(float(item / total))
        # best_index = probability_zones.index(max(probability_zones))
        # measured_zone_1 = zones[best_index]
        # probability_zones[best_index] = 0.0
        # measured_zone_2 = zones[probability_zones.index(max(probability_zones))]
        return measured_zone_1
    else:
        ap_rssi_dict = get_data_ap_combination(data, *normalized.access_points)
        location_method = get_calculation_function(location_mode)
        if location_mode == "NNv4" or location_mode == "kNNv2" or location_mode == "kNNv1":
            calculated_co_ordinate = location_method(centroid_points=centroids, rssis=ap_rssi_dict)
        if location_mode == "kNNv3":
            calculated_co_ordinate = location_method(grid_points=grid_points, rssis=ap_rssi_dict)
        # coord = get_NNv4_RSSI(centroid_points=centroids, rssis=ap_rssi_dict)
        calculated_zone = get_zone(zones=zones, co_ordinate=calculated_co_ordinate)
        self_vector = normalized.get_vector(calculated_zone)
        self_zone = max(self_vector, key=self_vector.get)
        return self_zone


def other_test(centroids: List[Centroid], location_mode: str,
               zones: List[Zone], grid_points: List[GridPoint],
               data: Dict[AccessPoint, int], svm_list: List[svm_model],
               ):
    if location_mode == "SVM":
        test_predict = list()
        for svm in svm_list:
            p_labs, p_acc, p_vals = svm_predict(y=[], x=[list(data.values())],
                                                m=svm,
                                                options="-q")
            test_predict.append(p_labs[0])
        best_predict_zone = max(test_predict, key=test_predict.count)
        predicted_zone = zones[int(best_predict_zone) - 1]
        return predicted_zone
    else:
        location_method = get_calculation_function(location_mode)
        # calculated_co_ordinate = location_method(centroid_points=centroids, list_rssis=list_rssi)
        if location_mode == "NNv4" or location_mode == "kNNv2" or location_mode == "kNNv1":
            calculated_co_ordinate = location_method(centroid_points=centroids, rssis=data)
        if location_mode == "kNNv3":
            calculated_co_ordinate = location_method(grid_points=grid_points, rssis=data)
        # coord = get_NNv4_RSSI(centroid_points=centroids, rssis=ap_rssi_dict)
        calculated_zone = get_zone(zones=zones, co_ordinate=calculated_co_ordinate)
        return calculated_zone


def filter_scan_data(scan_data,
                     access_points: List[AccessPoint],
                     zones: List[Zone]):
    samples = list()  # type: List[Sample]
    longest_rssis = 0
    bssid_rssi_dict = dict()  # type: Dict[AccessPoint, List[int]]
    for each_data in scan_data:
        BSSID = each_data['BSSID']
        RSSI = each_data['RSSIs']
        for access_point in access_points:
            if BSSID == access_point.id:
                bssid_rssi_dict[access_point] = RSSI
                if len(RSSI) > longest_rssis:
                    longest_rssis = len(RSSI)
    for index in range(longest_rssis):
        ap_rssi_dict = dict()  # type: Dict[AccessPoint, int]
        for key, rssis in bssid_rssi_dict.items():
            try:
                ap_rssi_dict[key] = rssis[index]
            except IndexError:
                # Hit because this AP may not have enough RSSI values. Append the average.
                ap_rssi_dict[key] = round(sum(rssis) / len(rssis))
        samples.append(Sample(zones[0], ap_rssi_dict))

    return samples


def get_data_ap_combination(data: Dict[AccessPoint, int], *access_points: AccessPoint) -> Dict[AccessPoint, int]:
    return {k: v for k, v in data.items() if k in access_points}


def convert_epoch_to_datetime(epoch_time):
    return datetime.fromtimestamp(epoch_time / 1e3).strftime("%m %d, %Y, %H:%M:%S")


@app.route('/offline', methods=['Get', 'POST', 'PUT'])
def offline():
    if request.method == 'PUT':
        sound(300, 1)
        dict_data = ast.literal_eval(request.form.getlist('PutData')[0])  # Gets the actual JSON data that was sent.
        print(dict_data)
        insert_floor = dict_data['Floor']
        insert_zone = dict_data['Zone']
        target_building = buildings[1]
        target_floor = target_building.floors[int(insert_floor)]
        target_ap = target_floor.access_points
        scan_data = dict_data["Scans"]["data"]
        filtered_data = filter_data(scan_data, target_ap)
        save_to_db(filtered_data, insert_floor, insert_zone)
        return json.dumps({"result": "Saved"})


def filter_data(data, access_points):
    ap_list = [ap.id for ap in access_points]
    filtered_data = dict()
    data.sort(key=lambda k: len(k.get('RSSIs')))
    data1 = sorted(data, key=lambda k: len(k.get('RSSIs')), reverse=True)
    for d in data1:
        if d['BSSID'] in ap_list:
            filtered_data[d['BSSID']] = d['RSSIs']
    return filtered_data


def save_to_db(scan_data, insert_floor, insert_zone):
    insert_position = "floors.{}.Data.{}.zone_data".format(insert_floor, insert_zone)
    data_updated = mongo.db.Data.update(
        {"building_name": "HOME"}
        , {"$push": {insert_position: scan_data}})
    floor_data = list(mongo.db.Data.find())


@app.route('/update', methods=['Get', 'POST', 'PUT'])
def update():
    global zone_data
    if request.method == 'PUT':
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
        get_annotation(data[0], "Unknown")
        data = {"Zone": str(data[1])}
        return json.dumps(data)

    if zone_data:
        return render_template("location.html", zone_data=zone_data)
    else:
        return render_template("location.html")


@app.route('/save_rssis', methods=['PUT'])
def save_rssis():
    # Beep me
    frequency = 4000
    duration = 1000
    winsound.Beep(frequency, duration)

    dict_data = ast.literal_eval(request.form.getlist('PutData')[0])  # Gets the actual JSON data that was sent.
    print(dict_data)  # Print to CLI just in case.
    scan_data = dict_data['Scans']['data']
    scan_num = dict_data['Scans']['Scan_number']
    sorted_data = count_data(scan_data, scan_num)
    print(sorted_data)
    response = store_data(dict_data)
    print(str(response))
    return json.dumps(response)


def count_data(scan_data, scan_number):
    scan_data = [d for d in scan_data if len(d['RSSIs']) == scan_number]
    sorted_data = sorted(scan_data, key=lambda k: mean(k.get('RSSIs')), reverse=True)
    return sorted_data


def update_annotation(broadcaster_data):
    building = broadcaster_data['Building']
    floor = broadcaster_data['Floor']
    zone = broadcaster_data['Zone']
    room = broadcaster_data['Room']
    building_img = cv2.imread('static/img/' + building)
    floor_img = cv2.imread('static/img/' + floor)
    room_img = cv2.imread('static/img/' + room)


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
        broadcaster_data = {
            'Actual_Zone': "Unknown",
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
            broadcaster_data = {'Index': i + 1,
                                'Actual_Zone': str(zone_actual),
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


def train_db(num_combination: int,
             selection_mode: str,
             train_mode: str,
             test_mode: str,
             type_mode: str,
             num_splits: int
             ):
    for building in buildings:
        for floor in building.floors:
            print("This is " + train_mode + " - " + test_mode)

            if type_mode == "AP":
                access_points = [access_point for access_point in floor.access_points if
                                 access_point.type == type_mode]
            elif type_mode == "Beacon":
                if floor.floor_id == "1":
                    return
                else:
                    access_points = [access_point for access_point in floor.access_points if
                                     access_point.type == type_mode]
            else:
                access_points = floor.access_points

            centroids = floor.get_centroids
            grid_points = floor.grid_points

            raw_data_1 = floor.data
            zones = floor.zones

            # Set mode:
            NormalizedMatrix.error_mode = "MAX"
            NormalizedMatrix.combination_mode = "WGT"
            NormalizedMatrix.type_mode = type_mode
            NormalizedMatrix.train_mode = train_mode
            NormalizedMatrix.test_mode = test_mode

            access_point_combinations = get_ap_combinations(access_points)  # type: List[Tuple[AccessPoint, ...]]
            divide_data = random_data(floor, 0.5, raw_data_1)
            testing_data = divide_data[0]
            training_data = divide_data[1]
            shuffle(training_data)
            shuffle(testing_data)
            print("-- Online data size: {}".format(len(testing_data)))
            print("-- Instantiated test lists.")
            print("-- Offline data size: {}".format(len(training_data)))

            test_results = KFold.create_kflod_combination(train_mode=train_mode,
                                                          training_data=training_data,
                                                          centroids=centroids,
                                                          grid_points=grid_points,
                                                          num_combination=num_combination,
                                                          zones=zones,
                                                          access_points=access_points,
                                                          num_splits=num_splits,
                                                          selection_mode=selection_mode)
            final_correct = dict()
            test_features = list()  # type: List[Dict[AccessPoint, int]]
            test_classes = list()  # type: List[int]
            for sample in testing_data:
                test_features.append(sample.scan)
                test_classes.append(sample.answer.num)
            if test_mode == "SVM":
                for ap, model in test_results.items():
                    svm_list = model[1]
                    test_predict = list()
                    correct = 0
                    aps_being_used = [x for x in ap]
                    ap_test_features = list()
                    for feature_set in test_features:
                        ap_test_features.append(
                            [value for key, value in feature_set.items() if key in aps_being_used])
                    for svm in svm_list:
                        p_labs, p_acc, p_vals = svm_predict(y=test_classes, x=ap_test_features, m=svm,
                                                            options="-q")
                        test_predict.append(p_labs)
                    for d in range(len(test_predict[0])):
                        zone_predictions = list()
                        for p in test_predict:
                            zone_predictions.append(p[d])
                        best_predict_zone = max(zone_predictions, key=zone_predictions.count)
                        predicted_zone = zones[int(best_predict_zone) - 1]
                        if predicted_zone.num == test_classes[d]:
                            correct += 1
                    gd_accuracy = correct / len(testing_data)
                    final_correct[ap] = gd_accuracy
            else:
                for ap_tuple, model in test_results.items():
                    normalized_matrix = model[0]
                    correct = 0
                    combination_method = get_combination_function('WGT')
                    for sample in testing_data:
                        zone_list = find_position(normalized_matrix, centroids,
                                                  zones, test_mode, grid_points,
                                                  sample.scan, combination_method)
                        if zone_list == sample.answer:
                            correct += 1
                    gd_accuracy = correct / len(testing_data)
                    final_correct[ap_tuple] = gd_accuracy
            best_set = max(final_correct.items(), key=operator.itemgetter(1))[0]
            floor.matrix = test_results[best_set][0]
            floor.model = test_results[best_set][1]


# region Main
if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0', port=5000)
# endregion
