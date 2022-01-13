import csv
import math
import math
import os
import random
import re
from collections import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from itertools import combinations
from random import shuffle
from statistics import mean
from time import time
from typing import List, Dict

import numpy as np
import pandas as pd
from celery import Celery
from flask import Flask
from flask_pymongo import PyMongo
from flask_socketio import SocketIO
from keras.layers import Dense
from keras.models import Sequential
from scipy.stats import normaltest, wilcoxon, mannwhitneyu
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import skfuzzy as fuzz
from Algorithms.SeparatingSVM.Model import Partition, MLModel
from Objects.LineBasedAlgorithm import LineBasedAlgorithm
from Resources.Objects.Building import Building
from Resources.Objects.Points.GridPoint_RSSI import GridPoint, get_gp_num
from Resources.Objects.TestData import Sample
from Resources.Objects.Zone import Zone

ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}
executor = ThreadPoolExecutor(2)
basePath = os.path.dirname(__file__)
position_data = list()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def random_data(zones: List[Zone], online_pct: float, samples: List[Sample]):
    online_samples = list()
    offline_samples = list()
    for zone in zones:
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

data = list(mongo.db.Data.find())
start_building_time = time()
buildings = Building.create_building_list(data)
end_building_time = time()
print("Building was created: {}s.\n".format(end_building_time - start_building_time))

floor = buildings[0].floors[0]
access_points = floor.access_points
filter_aps = [access_points[0], access_points[2], access_points[4], access_points[5]]
grid_points = floor.grid_points
raw_data = floor.data
ran_data = floor.random_data
zones = floor.zones
centroids = floor.get_centroids
grid_points_only = list()
zone_list = [zones[0], zones[4], zones[7], zones[10], zones[14]]
corner_gps = [grid_points[24], grid_points[28], grid_points[34], grid_points[38]]
# corner_gps = [grid_points[0], grid_points[3], grid_points[18], grid_points[21]]
# corner_gps = [grid_points[0], grid_points[5], grid_points[18], grid_points[23]]
for cp in centroids:
    for cnp in cp.CornerPoints:
        if cnp.num in [p.num for p in grid_points_only]:
            continue
        else:
            grid_points_only.append(cnp)

center_inputs = list()
center_targets = list()
random_inputs = list()
random_targets = list()
random_coords = list()
answer_pattern = list()
for sample in raw_data:
    center_targets.append(sample.answer.num)
    center_inputs.append([value for key, value in sample.scan.items()])

for sample in ran_data:
    random_coords.append(sample.coord)
    random_targets.append(sample.answer.num)
    random_inputs.append([value for key, value in sample.scan.items()])
#
# center_features = list()
# for feature_set in center_inputs:
#     center_features.append(
#         [value for key, value in feature_set.items()])
#
# random_features = list()
# for feature_set in random_inputs:
#     random_features.append(
#         [value for key, value in feature_set.items()])

# X_train, X_test, y_train, y_test = train_test_split(center_features, center_targets, test_size=0.2, random_state=42)
# X_train1, X_test1, y_train1, y_test1 = train_test_split(random_features, random_coords, test_size=0.8,
#                                                         random_state=42)
filter_data = list()
for i in range(len(ran_data)):
    sample = ran_data[i]
    coord = sample.coord
    if 1 < coord[1] < 9 and 1 < coord[0] < 5:
        filter_data.append(sample)
z_list = [1, 5, 8, 11, 15]
gp_data = [d for d in raw_data if d.answer.num in z_list]

X_center, Y_center = train_test_split(raw_data, test_size=0.05, random_state=42)
X_random, Y_random = train_test_split(filter_data, test_size=0.2, random_state=42)
X_train, X_test, X_filter1, y_train, y_test, X_filter2, X_corner, y_corner = ([] for i in range(8))
for sample in X_center:
    y_train.append(sample.answer.num)
    X_train.append([value for key, value in sample.scan.items()])
    X_filter1.append([value for key, value in sample.scan.items() if key in filter_aps])
for sample in Y_center:
    y_test.append(sample.answer.num)
    X_test.append([value for key, value in sample.scan.items()])
    X_filter2.append([value for key, value in sample.scan.items() if key in filter_aps])
for sample in gp_data:
    y_corner.append(sample.answer.num)
    X_corner.append([value for key, value in sample.scan.items()])

X_train2, X_test2, X_filter3, y_train2, y_test2, X_filter4, coord_train, coord_test = ([] for i in range(8))
for sample in X_random:
    coord_train.append(sample.coord)
    y_train2.append(sample.answer.num)
    X_train2.append([value for key, value in sample.scan.items()])
    X_filter3.append([value for key, value in sample.scan.items() if key in filter_aps])
for sample in Y_random:
    coord_test.append(sample.coord)
    y_test2.append(sample.answer.num)
    X_test2.append([value for key, value in sample.scan.items()])
    X_filter4.append([value for key, value in sample.scan.items() if key in filter_aps])
# X_train2, X_test2, y_train2, y_test2 = train_test_split(random_features, random_coords, test_size=0.2,
#                                                         random_state=42)

offline_X = X_train + X_train2
offline_Y = y_train + y_train2
online_X = X_test + X_test2
online_Y = y_test + y_test2
filter_train = X_filter1 + X_filter3
filter_test = X_filter2 + X_filter4
current_user = dict()
# normalized_matrices = List[NormalizedMatrix]
zone_data = []
position_data = []
location_data = []
room_list = []
models = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]


def sort_gps(gps):
    for i in range(len(gps)):
        min_index = i
        for j in range(i + 1, len(gps)):
            if gps[min_index] < gps[j]:
                min_index = j

        gps[i], gps[min_index] = gps[min_index], gps[i]
    return gps


def Nelson_method(sample, pointFrom, pointTo):
    letter_dict = {'6': 'A', '24': 'B', '1': 'C', '19': 'D', '32': 'E'}
    for gp in corner_gps:
        distances = 0
        for ap, rssi in sample.scan.items():
            distance = math.pow((rssi - gp.get_rssis(ap)), 2)
            distances += distance
        try:
            distances = math.sqrt(distances)
            gp.distance = distances
            # gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print("Oop!")

    sorted_gps = sort_gps(corner_gps)

    best_gps = sorted_gps
    point_list = [gp.point for gp in best_gps]
    centroid = grid_points[30]
    centroidSX = centroid.x
    centroidSY = centroid.y

    closestG1 = best_gps[0]
    closestG2 = best_gps[1]
    closestG2Alt = best_gps[2]

    midpointX = (closestG1.x + closestG2.x) / 2
    midpointY = (closestG1.y + closestG2.y) / 2

    if (midpointX == centroidSX and midpointY == centroidSY):
        # print("Switch")
        closestG2 = closestG2Alt
        midpointX = (closestG1.x + closestG2.x) / 2
        midpointY = (closestG1.y + closestG2.y) / 2

    # midpoint = GridPoint("0", 0, {}, midpointX, midpointY)

    # print("Centroid:{}, grid_point:{}, mid_point:{}".format(centroid.point, closestG1.point, midpoint.point))
    lba = LineBasedAlgorithm()
    resultingPoints = list()
    closestPoints = list()
    for i in range(10):
        shiftPercent = i * 10
        closestPoints.append(
            [letter_dict[str(closestG1.num)], letter_dict[str(closestG2.num)], letter_dict[str(centroid.num)]])
        if pointFrom == 's' and pointTo == 'm':
            resultingPoint = lba.compute_lba(centroid, closestG1, [closestG2], shiftPercent, True)
            resultingPoints.append(resultingPoint)

        elif pointFrom == 's' and pointTo == 'g':
            resultingPoint = lba.compute_lba(centroid, closestG1, [closestG2], shiftPercent, False)
            resultingPoints.append(resultingPoint)

        elif pointFrom == 'm' and pointTo == 's':
            resultingPoint = lba.compute_lba_mid(centroid, closestG1, [closestG2], shiftPercent, False)
            # runtime = timeit.default_timer() - starttime
            # print(runtime)
            resultingPoints.append(resultingPoint)

        elif pointFrom == 'm' and pointTo == 'g':
            resultingPoint = lba.compute_lba_mid(centroid, closestG1, [closestG2], shiftPercent, True)
            resultingPoints.append(resultingPoint)

        elif pointFrom == 'g' and pointTo == 's':
            resultingPoint = lba.compute_lba_grid(centroid, closestG1, [closestG2], shiftPercent, False)
            resultingPoints.append(resultingPoint)

        elif pointFrom == 'g' and pointTo == 'm':
            resultingPoint = lba.compute_lba_grid(centroid, closestG1, [closestG2], shiftPercent, True)
            resultingPoints.append(resultingPoint)

        # Otherwise, we have to move multiple points.
        else:
            resultingPoint = lba.compute_lba_multiple(centroid, closestG1, [closestG2], shiftPercent, pointTo)
            resultingPoints.append(resultingPoint)
    if resultingPoints[0] == resultingPoints[1]:
        print(resultingPoints[0])
    return resultingPoints, closestPoints


def calculateLineAlgorithmV4(sample):
    for gp in centroids:
        distances = 0
        for ap, rssi in sample.scan.items():
            distance = math.pow((rssi - gp.Center.get_rssis(ap)), 2)
            distances += distance
        try:
            distances = math.sqrt(distances)
            gp.Center.distance = distances
            # gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print("Oop!")

    sorted_cps = sort_gps([c.Center for c in centroids])

    for c in centroids:
        if c.Center.num == sorted_cps[0].num:
            closest_c = c

    for gp in closest_c.CornerPoints:
        distances = 0
        for ap, rssi in sample.scan.items():
            distance = math.pow((rssi - gp.get_rssis(ap)), 2)
            distances += distance
        try:
            distances = math.sqrt(distances)
            gp.distance = distances
            # gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print("Oop!")

    sorted_gps = sort_gps(closest_c.CornerPoints)
    closestG1 = sorted_gps[0]
    closestG2 = sorted_gps[1]
    closestG2Alt = sorted_gps[2]

    Ep = 0
    Eq = 0
    # P is the centroid and q is the grid point. Rp and Rq are offline measurements.
    for aIndex in range(0, len(access_points)):
        currentAP = access_points[aIndex]
        # print(currentAP.num - 1)
        Mp = sample.rssi(currentAP)
        Mq = sample.rssi(currentAP)
        Rp = closest_c.Center.get_rssis(currentAP)
        Rq = closestG1.get_rssis(currentAP)
        # print("MP:", Mp, "RP:", Rp)
        # print("MQ:", Mq, "RQ:", Rq)

        # Ep += 100 * abs(Mp - Rp) / Rp
        # Eq += 100 * abs(Mq - Rq) / Rq
        Ep += 10 ** (-1 * abs(Mp - Rp) / Rp)
        Eq += 10 ** (-1 * abs(Mq - Rq) / Rq)

        # print("EP:", Ep, "EQ:", Eq)
    # print("MP:", Mp, "RP:", Rp)
    # Use the formula

    m = Ep / len(access_points)
    n = Eq / len(access_points)

    # print("M:", m, "N:", n)
    x1 = closest_c.Center.x
    x2 = closestG1.x
    y1 = closest_c.Center.y
    y2 = closestG1.y

    # RX and RY are the new coordinates of the grid point.
    mn = (m + n)
    if (mn == 0.0):
        mn = 0.00000000001

    rx = (m * x2 + n * x1) / (mn)
    ry = (m * y2 + n * y1) / (mn)
    # print("Gx:", x2)
    # print("Gy:", y2)
    # print("Cx:", x1)
    # print("Cy:", y1)
    # print("Rx:", rx)
    # print("Ry:", ry)

    # Calculate the centroid between the 4 points.
    centroidSX = closest_c.Center.x
    centroidSY = closest_c.Center.y

    # Get the mid point.
    midpointX = (closestG1.x + closestG2.x) / 2
    midpointY = (closestG1.y + closestG2.y) / 2

    # print(firstClosestIndex)
    # print(secondClosestIndex)
    # Check to see if the two closest points are parallel to each other or not.
    if (midpointX == centroidSX and midpointY == centroidSY):
        # print("Switch")
        closestG2 = closestG2Alt
        midpointX = (closestG1.x + closestG2.x) / 2
        midpointY = (closestG1.y + closestG2.y) / 2

    # This code assumes that we are only using the grid point and centroid.

    # Now, we use the distance calculated.
    centroidQX = (rx + centroidSX + midpointX) / 3
    centroidQY = (ry + centroidSY + midpointY) / 3

    return [centroidQX, centroidQY]


def is_between(a, b, c):
    area = (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2
    if area == 0:
        return True
    return False


def findKNNRSSI(test_sample):
    test_gps = corner_gps + [grid_points[30]]
    for gp in test_gps:
        distances = 0
        for ap, rssi in test_sample.scan.items():
            distance = math.pow((rssi - gp.get_rssis(ap)), 2)
            distances += distance
        try:
            distances = math.sqrt(distances)
            gp.distance = distances
            # gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print("Oop!")

    sorted_points = sort_gps(corner_gps)
    centroid = grid_points[30]
    closestG1 = sorted_points[0]
    closestG2 = sorted_points[1]
    midpointX = (closestG1.x + closestG2.x) / 2
    midpointY = (closestG1.y + closestG2.y) / 2
    if (midpointX == centroid.x and midpointY == centroid.y):
        closestG2 = sorted_points[2]
    cloest_3 = [centroid, closestG1, closestG2]
    test_rssi = list(test_sample.scan.values())

    normalized_dict = dict()
    average_normalized = [0, 0, 0, 0, 0, 0]
    dot_dict = dict()
    for point in cloest_3:
        p_rssis = point.rssi
        rssi_list = list()
        average_dot = list()
        for i in range(len(p_rssis)):
            normalized_value = round(abs((test_rssi[i] - p_rssis[i]) / p_rssis[i]), 3)
            dot_value = round(math.pow(normalized_value, 2), 3)
            rssi_list.append(normalized_value)
            average_normalized[i] += normalized_value
            average_dot.append(dot_value)
        normalized_dict[point.num] = rssi_list
        dot_dict[point.num] = average_dot
    average_normalized = [round(a * 100, 3) for a in average_normalized]
    return normalized_dict, cloest_3, average_normalized, dot_dict


def determine_points(test_sample):
    in_line = [[1, 22], [4, 19], [22, 1], [19, 4]]
    test_gps = corner_gps + [grid_points[30]]
    average_normalized = [0, 0, 0, 0, 0, 0]
    center_list = list()
    dot_dict = list()
    test_rssi = list(test_sample.scan.values())
    for point in test_gps:
        p_rssis = point.rssi
        average_dot = list()
        rssi_list = list()
        distances = 0
        for ap, rssi in test_sample.scan.items():
            distance = math.pow((rssi - point.get_rssis(ap)), 2)
            distances += distance
        distances = round(math.sqrt(distances), 3)
        for i in range(len(p_rssis)):
            normalized_value = round(abs((test_rssi[i] - p_rssis[i]) / p_rssis[i]), 3)
            rssi_list.append(normalized_value)
            dot_value = round(math.pow(normalized_value, 2), 3)
            average_dot.append(dot_value)
        if point.num == grid_points[30].num:
            center_list += [distances, round(sum(average_dot), 3), point, rssi_list]
        else:
            dot_dict.append([distances, round(sum(average_dot), 3), point, rssi_list])
    dot_dict = sorted(dot_dict, key=lambda k: k[0], reverse=True)
    best_2 = [d[2].num for d in dot_dict[-2:]]
    if best_2 in in_line:
        all_points = [dot_dict[-1], dot_dict[-3], center_list]
    else:
        all_points = dot_dict[-2:] + [center_list]
    all_points = sorted(all_points, key=lambda kv: kv[0], reverse=True)
    point_list = [d[2] for d in all_points]

    letter_optimum = tuple([letter_dict[p.id] for p in point_list])
    for dot in all_points:
        p_rssis = dot[2].rssi
        for i in range(len(p_rssis)):
            normalized_value = round(abs((test_rssi[i] - p_rssis[i]) / p_rssis[i]), 3)
            average_normalized[i] += normalized_value
    average_normalized = [int(a * 100) for a in average_normalized]
    return point_list, letter_optimum, average_normalized, all_points


def find_closet(coord_list, letter_dict):
    closet_list = list()
    for c in coord_list:
        actual_x = c[0]
        actual_y = c[1]

        point_distance = dict()
        for z in zone_list:
            point = z.points.Center
            p_x = point.x
            p_y = point.y
            distance = math.sqrt(
                math.pow((p_x - actual_x), 2) + math.pow((p_y - actual_y), 2))
            point_distance[z.num] = distance
        sorted_distance = dict(sorted(point_distance.items(), key=lambda kv: kv[1], reverse=False))
        closet_point = list(sorted_distance.keys())[0]
        closet_list.append(letter_dict[closet_point])
    return closet_list


def findNormalizedRSSI(train_rssi, actual_coord):
    in_line = [[25, 39], [29, 35], [39, 25], [35, 29]]
    # in_line = [[1, 22], [4, 19], [22, 1], [19, 4]]
    # in_line = [[1, 24], [6, 19], [24, 1], [19, 6]]
    actual_x = actual_coord[0]
    actual_y = actual_coord[1]
    test_gps = corner_gps + [grid_points[31]]
    dot_list = list()
    dot_dict = list()
    average_normalized = [0, 0, 0, 0, 0, 0]
    center = list()
    for i in range(len(test_gps)):
        point = test_gps[i]
        p_rssis = point.rssi
        p_x = point.x
        p_y = point.y
        distance = math.sqrt(
            math.pow((p_x - actual_x), 2) + math.pow((p_y - actual_y), 2))
        average_dot = list()
        rssi_list = list()
        for i in range(len(p_rssis)):
            normalized_value = round(abs((train_rssi[i] - p_rssis[i]) / p_rssis[i]), 3)
            rssi_list.append(normalized_value)
            average_normalized[i] += normalized_value
            dot_value = round(math.pow(normalized_value, 2), 3)
            average_dot.append(dot_value)

        if point.id == '32':
            center = [distance, round(sum(average_dot), 3), point, rssi_list]
        else:
            dot_dict.append([distance, round(sum(average_dot), 3), point, rssi_list])
    sorted_average_dot = sorted(dot_dict, key=lambda kv: kv[1], reverse=True)
    closest_3 = sorted_average_dot[-2:]
    if [p[2].num for p in closest_3] in in_line:
        closest_3 = [sorted_average_dot[-1], sorted_average_dot[-3]]
    closest_3.append(center)
    closest_3 = sorted(closest_3, key=lambda kv: kv[1], reverse=True)
    closest_points = [p[2] for p in closest_3]
    letter_optimum = tuple([letter_dict[p.id] for p in closest_points])
    average_normalized = [int(a * 100) for a in average_normalized]

    return letter_optimum, average_normalized, closest_3


def calculateLinearApproach(sorted_points, actual_coord):
    closestG1 = sorted_points[0]
    closestG2 = sorted_points[1]
    closestG3 = sorted_points[2]
    # combo = [closestG1, closestG2, closestG3]
    combo_list = [(closestG1, closestG2, closestG3), (closestG1, closestG3, closestG2),
                  (closestG2, closestG3, closestG1)]
    optimum_percentage = 0
    optimum_accuracy = 100
    optimum_points = list()
    for combo in combo_list:
        x0 = combo[0].x
        y0 = combo[0].y

        x1 = combo[1].x
        y1 = combo[1].y

        x2 = combo[-1].x
        y2 = combo[-1].y
        for m in range(1, 10):
            n = 10 - m
            mn = m + n
            rx1 = (m * x2 + n * x0) / mn
            ry1 = (m * y2 + n * y0) / mn

            rx2 = (m * x2 + n * x1) / mn
            ry2 = (m * y2 + n * y1) / mn

            pred_x = (rx1 + rx2 + x2) / 3
            pred_y = (ry1 + ry2 + y2) / 3

            pred_distance = math.sqrt(
                math.pow((pred_x - actual_coord[0]), 2) + math.pow((pred_y - actual_coord[1]), 2))
            if pred_distance < optimum_accuracy:
                optimum_accuracy = pred_distance
                optimum_percentage = "{}-{}-{}-{}".format(combo[0].num, combo[1].num, combo[2].num, m)
                optimum_points = sorted([combo[0].num, combo[1].num, combo[2].num])
    return tuple(optimum_points), optimum_percentage


def calculateTwoShift(best_3, actual_coord, letter_point):
    combo = [letter_point[d] for d in best_3]
    optimum_accuracy = 100
    accuracy_list = dict()
    shift = 0
    x0 = combo[0].x
    y0 = combo[0].y

    x1 = combo[1].x
    y1 = combo[1].y

    x2 = combo[2].x
    y2 = combo[2].y
    for m in range(0, 10):
        rm = 10 - m
        rx1 = (m * x2 + rm * x0) / 10
        ry1 = (m * y2 + rm * y0) / 10
        for n in range(0, 10):
            rn = 10 - n
            rx2 = (n * x2 + rn * x1) / 10
            ry2 = (n * y2 + rn * y1) / 10

            pred_x = round(((rx1 + rx2 + x2) / 3), 1)
            pred_y = round(((ry1 + ry2 + y2) / 3), 1)

            pred_distance = math.sqrt(
                math.pow((pred_x - actual_coord[0]), 2) + math.pow((pred_y - actual_coord[1]), 2))
            accuracy_list[(m, n)] = [round(pred_distance, 3), (pred_x, pred_y)]
            if pred_distance < optimum_accuracy:
                optimum_accuracy = round(pred_distance, 3)
                shift = (m, n)
    # accuracy_list = sorted(accuracy_list, key=lambda kv: kv[1], reverse=False)
    return shift, accuracy_list


def calculateJSApproach(best_3, actual_coord, letter_point):
    combo = [letter_point[d] for d in best_3]
    combo_list = [combo]
    optimum_accuracy = 100
    accuracy_list = list()
    shift = 0
    for combo in combo_list:
        x0 = combo[0].x
        y0 = combo[0].y

        x1 = combo[1].x
        y1 = combo[1].y

        x2 = combo[2].x
        y2 = combo[2].y
        for m in range(0, 10):
            n = 10 - m
            mn = m + n
            rx1 = (m * x2 + n * x0) / mn
            ry1 = (m * y2 + n * y0) / mn

            rx2 = (m * x2 + n * x1) / mn
            ry2 = (m * y2 + n * y1) / mn

            pred_x = round(((rx1 + rx2 + x2) / 3), 1)
            pred_y = round(((ry1 + ry2 + y2) / 3), 1)

            pred_distance = math.sqrt(
                math.pow((pred_x - actual_coord[0]), 2) + math.pow((pred_y - actual_coord[1]), 2))
            accuracy_list.append([m, round(pred_distance, 3), (pred_x, pred_y)])
            if pred_distance < optimum_accuracy:
                optimum_accuracy = round(pred_distance, 3)
                shift = m
    # accuracy_list = sorted(accuracy_list, key=lambda kv: kv[1], reverse=False)
    return shift, accuracy_list


def get_two_coord(point_list, shifts):
    x0, y0 = point_list[0].point
    x1, y1 = point_list[1].point
    x2, y2 = point_list[2].point
    m = shifts[0]
    rm = 10 - m
    n = shifts[1]
    rn = 10 - n
    rx1 = (m * x2 + rm * x0) / 10
    ry1 = (m * y2 + rm * y0) / 10

    rx2 = (n * x2 + rn * x1) / 10
    ry2 = (n * y2 + rn * y1) / 10

    pred_x = round(((rx1 + rx2 + x2) / 3), 1)
    pred_y = round(((ry1 + ry2 + y2) / 3), 1)

    return [(pred_x, pred_y)]


def get_coord(point_list, optimum_data):
    coord_list = list()
    for d in optimum_data:
        x0, y0 = point_list[0].point
        x1, y1 = point_list[1].point
        x2, y2 = point_list[2].point
        m = int(d)
        n = 10 - m
        mn = m + n
        rx1 = (m * x2 + n * x0) / mn
        ry1 = (m * y2 + n * y0) / mn

        rx2 = (m * x2 + n * x1) / mn
        ry2 = (m * y2 + n * y1) / mn

        pred_x = round(((rx1 + rx2 + x2) / 3), 1)
        pred_y = round(((ry1 + ry2 + y2) / 3), 1)

        coord_list.append((pred_x, pred_y))
    return coord_list


def get_distance_error(test_coords, actual_coords):
    avg_distance = list()
    for i in range(len(actual_coords)):
        actual_coord = actual_coords[i]
        test_coord = test_coords[i]
        pred_distance = math.sqrt(
            math.pow((test_coord[0] - actual_coord[0]), 2) + math.pow((test_coord[1] - actual_coord[1]), 2))
        #     print(pred_distance)
        avg_distance.append(pred_distance)
    return avg_distance


def square_distance(x, y): return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def find_between_4(best_adjacent):
    # if len(best_adjacent) <= 3:
    #     return None
    # tuple1 = tuple(best_adjacent[:2])
    # tuple2 = tuple([best_adjacent[2]])
    tuple1 = tuple(best_adjacent)
    return {tuple1: None}


def calculate_margins(current_gp, rest_gps):
    current_num = [cp.num for cp in current_gp]
    margins = dict()
    for z in rest_gps:
        margin = binary_classification(z.num, current_num)
        margins[z.num] = margin
    margins = sorted(margins.items(), key=lambda kv: kv[1], reverse=False)
    return margins


def svm_c(k, observed_gps):
    if k == 1 and k == len(observed_gps):
        return None
    random_gps = random.sample(range(len(observed_gps)), k)
    partition_gps = list()
    for i in random_gps:
        partition_gps.append([observed_gps[i]])
    rest_gps = [z for index, z in enumerate(observed_gps) if index not in random_gps]
    partition_gps = svmc_partition(partition_gps, rest_gps)
    return partition_gps


def svmc_partition(partition_gps, rest_gps):
    margin_list = list()
    duplicate_list = list()
    for index, z in enumerate(partition_gps):
        margins = calculate_margins(z, rest_gps)
        margin_list.append(margins)
        duplicate_list.append(margins[0][0])

    while len([d for d in duplicate_list if d != 0]) != len(set([d for d in duplicate_list if d != 0])):
        for i in range(0, len(duplicate_list) - 1):
            zone_i = duplicate_list[i]
            value_i = margin_list[i][0][1]
            for j in range(i + 1, len(duplicate_list)):
                zone_j = duplicate_list[j]
                if zone_i == zone_j and zone_i != 0:
                    value_j = margin_list[j][0][1]
                    if value_i < value_j:
                        if len(margin_list[j]) > 1:
                            margin_list[j].pop(0)
                            duplicate_list[j] = margin_list[j][0][0]
                        else:
                            duplicate_list[j] = 0
                    else:
                        if len(margin_list[i]) > 1:
                            margin_list[i].pop(0)
                            duplicate_list[i] = margin_list[i][0][0]
                        else:
                            duplicate_list[i] = 0

    for index, d in enumerate(duplicate_list):
        if d != 0:
            partition_gps[index].append(get_gp_num(grid_points, d))

    occupied_gps = []
    for d in partition_gps:
        for s in d:
            occupied_gps.append(s.num)
    rest_gps = [z for z in grid_points if z.num not in occupied_gps]

    if rest_gps:
        svmc_partition(partition_gps, rest_gps)
    return partition_gps


def space_partition(partition_gps):
    empty_pattern = dict()
    gp_list = [gp.num for gp in partition_gps]
    # cp_list = [p.points.point for p in partition_gps]
    cp1, cp2 = find_furthest([gp.point for gp in partition_gps])
    partition1 = {cp1.num: 0}
    partition2 = {cp2.num: 0}
    best_adjacent1, best_adjacent2 = find_new_adjacent(partition1, partition2, gp_list)
    # best_adjacent1, best_adjacent2 = find_best_adjacent(partition1, partition2, gp_list)
    best_tuple1 = tuple(best_adjacent1.keys())
    best_tuple2 = tuple(best_adjacent2.keys())
    best_gps1 = [floor.find_gp(gp) for gp in best_adjacent1.keys()]
    best_gps2 = [floor.find_gp(gp) for gp in best_adjacent2.keys()]
    if len(best_adjacent1) > 3:
        pattern_1 = space_partition(best_gps1)
        empty_pattern[best_tuple1] = pattern_1
    else:
        # pattern_3 = find_between_4(list(best_adjacent1.keys()))
        empty_pattern[best_tuple1] = None
    if len(best_adjacent2) > 3:
        pattern_2 = space_partition(best_gps2)
        empty_pattern[best_tuple2] = pattern_2
    else:
        # pattern_4 = find_between_4(list(best_adjacent2.keys()))
        empty_pattern[best_tuple2] = None
    return empty_pattern


def find_new_adjacent(partition1, partition2, gps_list):
    adjacent1 = list()
    for ogz in list(partition1.keys()):
        gp = floor.find_gp(ogz)
        for a in list(gp.margin.keys()):
            a = int(a)
            if a not in adjacent1 and a not in list(partition1.keys()) and a in gps_list:
                adjacent1.append(a)

    margin_dict1 = dict()
    for b in adjacent1:
        margin = binary_classification(b, list(partition1.keys()))
        margin_dict1[b] = margin
    margin_dict1 = dict(sorted(margin_dict1.items(), key=lambda kv: kv[1], reverse=False))
    if len(adjacent1) == 0:
        print(margin_dict1)
    best_adjacent1 = list(margin_dict1.keys())[0]

    while best_adjacent1 != "":
        if best_adjacent1 in list(partition2.keys()):
            margin1 = margin_dict1[best_adjacent1]
            margin2 = partition2[best_adjacent1]
            if margin1 < margin2:
                partition1[best_adjacent1] = margin1
                partition2.pop(best_adjacent1)
                best_adjacent1 = ""
            else:
                margin_dict1.pop(best_adjacent1)
                if margin_dict1:
                    best_adjacent1 = list(margin_dict1.keys())[0]
                else:
                    best_adjacent1 = ""
        else:
            if best_adjacent1 != "":
                partition1[best_adjacent1] = margin_dict1[best_adjacent1]
                best_adjacent1 = ""

    adjacent2 = list()
    for ogz in list(partition2.keys()):
        gp = floor.find_gp(ogz)
        for a in list(gp.margin.keys()):
            a = int(a)
            if a not in adjacent2 and a not in list(partition2.keys()) and a in gps_list:
                adjacent2.append(a)
    margin_dict2 = dict()
    for c in adjacent2:
        margin = binary_classification(c, list(partition2.keys()))
        margin_dict2[c] = margin
    margin_dict2 = dict(sorted(margin_dict2.items(), key=lambda kv: kv[1], reverse=False))
    best_adjacent2 = list(margin_dict2.keys())[0]

    while best_adjacent2 != "":
        if best_adjacent2 in list(partition1.keys()):
            margin2 = margin_dict2[best_adjacent2]
            margin1 = partition1[best_adjacent2]
            if margin2 < margin1:
                partition2[best_adjacent2] = margin2
                partition1.pop(best_adjacent2)
                best_adjacent2 = ""
            else:
                partition1[best_adjacent2] = margin1
                margin_dict2.pop(best_adjacent2)
                if margin_dict2:
                    best_adjacent2 = list(margin_dict2.keys())[0]
                else:
                    best_adjacent2 = ""
        else:
            if best_adjacent2 != "":
                partition2[best_adjacent2] = margin_dict2[best_adjacent2]
                best_adjacent2 = ""
    partiton_zones = list(partition1.keys()) + list(partition2.keys())
    if set(partiton_zones) != set(gps_list):
        partition1, partition2 = find_new_adjacent(partition1, partition2, gps_list)

    return partition1, partition2


def find_best_adjacent(partition1, partition2, gps_list):
    adjacent1 = list()
    for ogz in list(partition1.keys()):
        gp = floor.find_gp(ogz)
        for a in list(gp.margin.keys()):
            a = int(a)
            if a not in adjacent1 and a not in list(partition1.keys()) and a in gps_list:
                adjacent1.append(a)

    margin_dict1 = dict()
    for b in adjacent1:
        margin = binary_classification(b, list(partition1.keys()))
        margin_dict1[b] = margin
    margin_dict1 = dict(sorted(margin_dict1.items(), key=lambda kv: kv[1], reverse=False))
    if len(adjacent1) == 0:
        print(margin_dict1)
    best_adjacent1 = list(margin_dict1.keys())[0]

    adjacent2 = list()
    for ogz in list(partition2.keys()):
        gp = floor.find_gp(ogz)
        for a in list(gp.margin.keys()):
            a = int(a)
            if a not in adjacent2 and a not in list(partition2.keys()) and a in gps_list:
                adjacent2.append(a)
    margin_dict2 = dict()
    for c in adjacent2:
        margin = binary_classification(c, list(partition2.keys()))
        margin_dict2[c] = margin
    margin_dict2 = dict(sorted(margin_dict2.items(), key=lambda kv: kv[1], reverse=False))
    best_adjacent2 = list(margin_dict2.keys())[0]
    while best_adjacent1 != "" or best_adjacent2 != "":
        if best_adjacent2 in list(partition1.keys()):
            margin2 = margin_dict2[best_adjacent2]
            margin1 = partition1[best_adjacent2]
            if margin2 < margin1:
                partition2[best_adjacent2] = margin2
                partition1.pop(best_adjacent2)
                best_adjacent2 = ""
            else:
                partition1[best_adjacent2] = margin1
                margin_dict2.pop(best_adjacent2)
                if margin_dict2:
                    best_adjacent2 = list(margin_dict2.keys())[0]
                else:
                    best_adjacent2 = ""

        elif best_adjacent1 in list(partition2.keys()):
            margin1 = margin_dict1[best_adjacent1]
            margin2 = partition2[best_adjacent1]
            if margin1 < margin2:
                partition1[best_adjacent1] = margin1
                partition2.pop(best_adjacent1)
                best_adjacent1 = ""
            else:
                margin_dict1.pop(best_adjacent1)
                if margin_dict1:
                    best_adjacent1 = list(margin_dict1.keys())[0]
                else:
                    best_adjacent1 = ""

        elif best_adjacent1 == best_adjacent2:
            margin1 = margin_dict1[best_adjacent1]
            margin2 = margin_dict2[best_adjacent2]
            if margin1 < margin2:
                partition1[best_adjacent1] = margin1
                margin_dict2.pop(best_adjacent2)
                if margin_dict2:
                    best_adjacent2 = list(margin_dict2.keys())[0]
                else:
                    best_adjacent2 = ""
                best_adjacent1 = ""
            else:
                partition2[best_adjacent2] = margin2
                margin_dict1.pop(best_adjacent1)
                if margin_dict1:
                    best_adjacent1 = list(margin_dict1.keys())[0]
                else:
                    best_adjacent1 = ""
                best_adjacent2 = ""
        else:
            if best_adjacent1 != "":
                partition1[best_adjacent1] = margin_dict1[best_adjacent1]
                best_adjacent1 = ""
            if best_adjacent2 != "":
                partition2[best_adjacent2] = margin_dict2[best_adjacent2]
                best_adjacent2 = ""

    # if len(margin_dict2) > 1 or len(margin_dict1) > 1:
    partiton_zones = list(partition1.keys()) + list(partition2.keys())
    if set(partiton_zones) != set(gps_list):
        partition1, partition2 = find_best_adjacent(partition1, partition2, gps_list)

    return partition1, partition2


def binary_classification(adjacent, compare_gps):
    bc_inputs = list()
    bc_targets = list()
    for sample in raw_data:
        sample_gp = sample.answer.num
        if sample_gp in compare_gps:
            bc_targets.append(0)
            bc_inputs.append([value for key, value in sample.scan.items()])
        if sample_gp == adjacent:
            bc_targets.append(1)
            bc_inputs.append([value for key, value in sample.scan.items()])
    svm = SVC(kernel='linear')
    svm.fit(bc_inputs, bc_targets)
    margin = 1 / np.sqrt(np.sum(svm.coef_ ** 2))
    return margin


def find_furthest(gps):
    max_square_distance = 0
    max_pair = tuple()
    for pair in combinations(gps, 2):
        if square_distance(*pair) > max_square_distance:
            max_square_distance = square_distance(*pair)
            max_pair = pair
    cp1 = find_gp(max_pair[0])
    cp2 = find_gp(max_pair[1])
    return cp1, cp2


def find_gp(coord):
    for gp in grid_points:
        if gp.point == coord:
            return gp


def test_tree(train_mode1, train_mode2, partitions, ap_test_feature):
    if partitions.left is not None:
        result = partitions.model.test_one(train_mode1, [ap_test_feature]).item(0)
        if result == 0:
            partitions = partitions.left
        else:
            partitions = partitions.right
        # if result < 2 and len(partitions.data) > 3:
        result = test_tree(train_mode1, train_mode2, partitions, ap_test_feature)
    elif len(partitions.data) == 1:
        result = partitions.data[0]
    else:
        result = partitions.model.test_one(train_mode2, [ap_test_feature]).item(0)
    return result


def get_closet_points(prob_results, labels):
    in_line = [['B', 'C'], ['C', 'B'], ['A', 'D'], ['D', 'A']]
    closet_list = list()
    for p in prob_results:
        best_points = dict()
        for i in range(len(p)):
            best_points[labels[i]] = p[i]
        point_e = best_points.pop('E')
        point_e = ['E', point_e]
        best_points = [[key, value] for key, value in best_points.items()]
        best_points = sorted(best_points, key=lambda k: k[1], reverse=False)
        best_3 = best_points[-2:]
        if [p[0] for p in best_3] in in_line:
            best_3 = [best_points[-1], best_points[-3]]
        best_3.append(point_e)
        best_3 = sorted(best_3, key=lambda kv: kv[1], reverse=False)
        closet_list.append([b[0] for b in best_3])
    return closet_list


def tree_build(train_mode1: str, train_mode2: str, pattern: Dict, gp_pattern: dict):
    # level = count(pattern)
    root = Partition([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    partitions = create_model(train_mode1, train_mode2, pattern, root, center_inputs, center_targets)
    # partitions.other_name()
    result_class = list()
    test_result = list()
    level_report = {0: 0, 1: 0, 2: 0, 3: 0}
    level_coords = list()
    level_4_num = 0
    last_right = 0
    last_level = 0
    sp_test_start = time()
    for n in range(len(random_inputs)):
        ap_test_feature = random_inputs[n]
        actual_result = random_targets[n]
        actual_level = gp_pattern[random_targets[n]]
        if len(actual_level) > 3:
            level_4_num += 1
        result = test_tree(train_mode1, train_mode2, partitions, ap_test_feature)

        level_result = gp_pattern[result]
        if level_result == actual_level:
            last_level += 1
            if actual_result == result:
                last_right += 1
        level_coord = gp_coord[result]
        level_coords.append(level_coord)
        for i in range(len(actual_level)):
            if actual_level[:i] == level_result[:i]:
                level_report[i] += 1
        test_result.append(get_gp_num(grid_points, result).point)
        result_class.append(result)
    # result = test_tree(partitions, X_test[0], 0, train_mode)
    for k in range(3):
        l_result = [l[k] for l in level_coords]
        l_report = get_distance_error(l_result, random_coords)
        print("{} level testing Accuracy: {}".format(k + 1, mean(l_report)))
    sp_test_end = time()
    sp_test_time = sp_test_end - sp_test_start
    report = get_distance_error(test_result, random_coords)
    print("H-SVM testing Time: {}".format(sp_test_time))
    # print(metrics.classification_report(random_targets, result_class))
    print(level_report)
    print("level4 Total：{}".format(level_4_num))
    print("H-SVM testing Accuracy: {}".format(mean(report)))
    print("level1:{}".format(level_report[0] / len(random_inputs)))
    print("level2:{}".format(level_report[1] / len(random_inputs)))
    print("level3:{}".format(level_report[2] / len(random_inputs)))
    print("level4:{}".format(level_report[3] / level_4_num))
    print("Last Right:{}".format(last_right / last_level))
    # plt.plot(report)
    print('\n')
    return report


def train_model(train_mode, features, classes, k=3):
    svm = SVC(kernel='rbf', decision_function_shape='ovr', gamma='scale')
    svm1 = SVC(kernel='rbf', decision_function_shape='ovr', gamma='scale')
    gnb = GaussianNB()
    rf = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors=k)
    dt = DecisionTreeClassifier()
    ann = Sequential()
    ann.add(Dense(12, activation='relu'))
    ann.add(Dense(8, activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))
    ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = MLModel(rf=rf, knn=knn, ann=ann, dt=dt, svm=svm, gnb=gnb, svm1=svm1)
    model.train_one(train_mode, features, classes)

    return model


def create_model(train_mode1, train_mode2, pattern, root, training_X, training_Y):
    if isinstance(pattern, dict):
        ap_list = list(pattern.keys())
        class_list = list()
        data_list = list()

        for num, d in enumerate(ap_list):
            class_row = [row for row, t in enumerate(training_Y) if t in d]
            data_list += [training_X[row] for row in class_row]
            class_list += [num for i in range(len(class_row))]

        model = train_model(train_mode1, data_list, class_list)
        root.model = model
        for key, val in pattern.items():
            p = Partition(list(key))
            root.insert(p)
            if isinstance(val, dict):
                create_model(train_mode1, train_mode2, val, p, training_X, training_Y)
            else:
                if len(key) > 1:
                    base_row = list()
                    base_list = list()
                    for row, t in enumerate(training_Y):
                        if t in key:
                            base_row.append(row)
                            base_list.append(t)
                    base_data = [training_X[row] for row in base_row]
                    model = train_model(train_mode2, base_data, base_list)
                    p.model = model

    return root


def find_center(gps):
    list_x = [gp.x for gp in gps]
    list_y = [gp.y for gp in gps]
    center_x = mean(list_x)
    center_y = mean(list_y)
    return (center_x, center_y)


def find_partition(target_class, pattern):
    for index, p in enumerate(pattern):
        if target_class in p:
            return index


def find_pattern(k, partition_pattern):
    p_list = list()
    if k == 2:
        p_list = list(partition_pattern.keys())
    else:
        for key, value in partition_pattern.items():
            for ki, v in value.items():
                if k == 4:
                    p_list.append(ki)
                    continue
                for a, b in v.items():
                    if k == 8:
                        p_list.append(a)
                        continue
    return p_list


def get_result(train_mode1, train_mode2, pattern, k, name):
    new_offline_Y = list()
    new_offline_X = list()
    model_list = list()
    for j in range(k):
        zone_nums = pattern[j]
        zone_X = list()
        zone_Y = list()
        if len(zone_nums) > 1:
            for i in range(len(center_inputs)):
                offline_row = center_inputs[i]
                offline_class = center_targets[i]
                if offline_class in zone_nums:
                    new_offline_Y.append(j)
                    new_offline_X.append(offline_row)
                    zone_X.append(offline_row)
                    zone_Y.append(offline_class)
            test_model = train_model(train_mode2, zone_X, zone_Y)
        else:
            test_model = zone_nums[0]
        model_list.append(test_model)

    compare_model = train_model(train_mode1, new_offline_X, new_offline_Y)
    svm_test_start = time()
    compare_results = list()
    right_classification = 0
    for x in range(len(random_inputs)):
        ap_test_feature = random_inputs[x]
        target_class = random_targets[x]
        target_partition = find_partition(target_class, pattern)
        compare_result = list(compare_model.test_one(train_mode1, [ap_test_feature]))[0]
        test_m = model_list[compare_result]
        if isinstance(test_m, int):
            fine_result = test_m
        else:
            fine_result = list(test_m.test_one(train_mode2, [ap_test_feature]))[0]
        if target_partition == compare_result:
            right_classification += 1
        # center = centers[compare_result]
        compare_results.append(get_gp_num(grid_points, fine_result).point)
    # compare_result = list(compare_model.test_one("svm1", [X_test[0]]))[0]
    svm_test_end = time()
    compare_report = get_distance_error(compare_results, random_coords)
    # compare_report = metrics.classification_report(y_test, compare_results)
    # compare_accuracy = metrics.accuracy_score(y_test, compare_results)
    svm_test_time = svm_test_end - svm_test_start
    print("{}-SVM-C testing Time: {}".format(train_mode1, svm_test_time))
    # print(confusion_matrix(y_test, compare_results))
    print("{}-{}-{} testing Accuracy: {}".format(train_mode1, train_mode2, name, mean(compare_report)))
    print("{}-{}-{} classification Accuracy: {}".format(train_mode1, train_mode2, name,
                                                        right_classification / len(random_inputs)))

    return compare_report


# def find_average(normalized_dict):


pattern = {
    (1, 2, 6, 7, 11, 12, 13): {
        (1, 2, 7): {
            (2, 7): None,
            (1,): None
        },
        (6, 11, 12, 13): {
            (12, 13): None,
            (6, 11): None
        }
    },
    (3, 4, 5, 8, 9, 10, 14, 15): {
        (3, 4, 5, 8): {
            (3, 8): None,
            (4, 5): None
        },
        (9, 10, 14, 15): {
            (9, 14): None,
            (10, 15): None
        }
    }
}

check_pattern = {
    (1, 2, 6, 7, 11, 12, 13): {
        (1, 2, 3, 4, 5, 7, 8): {
            (1, 2, 7): {
                (2, 7): None,
                (1,): None
            },
            (3, 4, 5, 8): {
                (3, 8): None,
                (4, 5): None
            }
        },
        (6, 11, 12, 13): {
            (12, 13): None,
            (6, 11): None
        }
    },
    (3, 4, 5, 8, 9, 10, 14, 15): {
        (1, 2, 3, 4, 5, 7, 8): {
            (1, 2, 7): {
                (2, 7): None,
                (1,): None
            },
            (3, 4, 5, 8): {
                (3, 8): None,
                (4, 5): None
            }
        },
        (9, 10, 14, 15): {
            (9, 14): None,
            (10, 15): None
        }
    }
}

check_pattern1 = {
    (1, 2, 6, 7, 11, 12, 13): {
        (6, 9, 10, 11, 12, 13, 14, 15): {
            (6, 11, 12, 13): {
                (6, 11): None,
                (12, 13): None
            },
            (9, 10, 14, 15): {
                (9, 14): None,
                (10, 15): None
            }
        },
        (1, 2, 7): {
            (2, 7): None,
            (1,): None
        }
    },
    (3, 4, 5, 8, 9, 10, 14, 15): {
        (6, 9, 10, 11, 12, 13, 14, 15): {
            (6, 11, 12, 13): {
                (6, 11): None,
                (12, 13): None
            },
            (9, 10, 14, 15): {
                (9, 14): None,
                (10, 15): None
            }
        },
        (3, 4, 5, 8): {
            (3, 8): None,
            (4, 5): None
        }
    }
}

pattern1 = {
    (1, 2, 3, 6, 7, 8, 11, 12): {
        (1, 2, 3, 8): {
            (1, 2): None,
            (3, 8): None
        },
        (6, 7, 11, 12): {
            (6, 7): None,
            (11, 12): None
        }
    },
    (4, 5, 9, 10, 13, 14, 15): {
        (4, 5, 9, 10): {
            (4, 5): None,
            (9, 10): None
        },
        (13, 14, 15): {
            (13, 14): None,
            (15,): None
        }
    }
}

pattern2 = {
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10): {
        (1, 2, 3, 6): {
            (1, 6): None,
            (2, 3): None
        },
        (4, 5, 7, 8, 9, 10): {
            (4, 5, 10): {
                (5, 10): None,
                (4,): None
            },
            (7, 8, 9): {
                (7, 8): None,
                (9,): None
            }
        }
    },
    (11, 12, 13, 14, 15): {
        (11, 12): None,
        (13, 14, 15): {
            (13, 14): None,
            (15,): None
        }
    }
}
pattern3 = {
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10): {
        (1, 2, 3, 6): {
            (1, 6): None,
            (2, 3): None
        },
        (4, 5, 7, 8, 9, 10): {
            (5, 10): None,
            (4, 7, 8, 9): {
                (7, 8): None,
                (4, 9): None
            }
        }
    },
    (11, 12, 13, 14, 15): {
        (11, 12): None,
        (13, 14, 15): {
            (13, 14): None,
            (15,): None
        }
    }
}
proposed_pattern = {
    (1, 6, 7, 11, 12, 17, 16, 2): {
        (1, 2, 7, 6): {
            (1, 2): None,
            (6, 7): None
        },
        (17, 12, 11, 16): {
            (11, 16): None,
            (12, 17): None
        }
    },
    (20, 15, 19, 14, 13, 9, 10, 4, 8, 3, 18, 5): {
        (20, 19, 14, 13, 15, 18): {
            (20, 18): None,
            (13, 14, 15, 19): None
        },
        (3, 4, 5, 9, 10, 8): {
            (3, 5): None,
            (8, 9, 10, 4): None
        }
    }
}

gp_pattern = {
    1: [0, 0, 0],
    2: [0, 0, 0],
    3: [1, 1, 0],
    4: [1, 1, 1, 0],
    5: [1, 1, 0],
    6: [0, 0, 1],
    7: [0, 0, 1],
    8: [1, 1, 1, 1],
    9: [1, 1, 1, 0],
    10: [1, 1, 1, 0],
    11: [0, 1, 1],
    12: [0, 1, 0],
    13: [1, 0, 1, 0],
    14: [1, 0, 1, 0],
    15: [1, 0, 1, 1],
    16: [0, 1, 1],
    17: [0, 1, 0],
    18: [1, 0, 0],
    19: [1, 0, 1, 0],
    20: [1, 0, 0]
}

gp_coord = {
    1: [(3.0, 1.0), (1.0, 1.0), (0, 0)],
    2: [(3.0, 1.0), (1.0, 1.0), (0, 2)],
    3: [(3.0, 6.0), (1.0, 6.0), (0, 4)],
    4: [(3.0, 6.0), (1.0, 6.0), (1.5, 6.0)],
    5: [(3.0, 6.0), (1.0, 6.0), (0, 8)],
    6: [(3.0, 1.0), (1.0, 1.0), (2, 0)],
    7: [(3.0, 1.0), (1.0, 1.0), (2, 2)],
    8: [(3.0, 6.0), (1.0, 6.0), (1.5, 6.0)],
    9: [(3.0, 6.0), (1.0, 6.0), (1.5, 6.0)],
    10: [(3.0, 6.0), (1.0, 6.0), (1.5, 6.0)],
    11: [(3.0, 1.0), (5.0, 1.0), (4, 0)],
    12: [(3.0, 1.0), (5.0, 1.0), (4, 2)],
    13: [(3.0, 6.0), (5.0, 6.0), (4.5, 6.0)],
    14: [(3.0, 6.0), (5.0, 6.0), (4.5, 6.0)],
    15: [(3.0, 6.0), (5.0, 6.0), (4.5, 6.0)],
    16: [(3.0, 1.0), (5.0, 1.0), (6, 0)],
    17: [(3.0, 1.0), (5.0, 1.0), (6, 2)],
    18: [(3.0, 6.0), (5.0, 6.0), (6, 4)],
    19: [(3.0, 6.0), (5.0, 6.0), (4.5, 6.0)],
    20: [(3.0, 6.0), (5.0, 6.0), (6, 8)],
}

level_4 = [4, 8, 9, 10, 13, 14, 15, 19]
level_5 = [4, 9, 10, 13, 14, 19]
# sp_test_start = time()
# test_result = list()
# for ap_test_feature in X_test:
#     result = test_tree(partitions, ap_test_feature, 0, train_mode)
#     test_result.append(result)
# accuracy = metrics.accuracy_score(y_test, test_result)
# # accuracy = metrics.classification_report(y_test, test_result)
# sp_test_end = time()
# sp_test_time = sp_test_end - sp_test_start
# print("SP-SVM testing Time: {}".format(sp_test_time))
# print(confusion_matrix(y_test, test_result))
# print("SP-SVM Testing Accuracy: {}".format(accuracy))
#

nelson_coord = list()
nelson_points = list()
module_A = list()

# for sample_data in Y_random:
#     # MAPoint = calculateLineAlgorithmV4(sample_data)
#     resultingPoint, closestPoints = Nelson_method(sample_data, 'mul', 's')
#     nelson_coord.append(resultingPoint)
#     nelson_points.append(closestPoints)
#     # module_A.append(MAPoint)
# for i in range(10):
#     nelson_d = list()
#     point_d = list()
#     for p in nelson_points:
#         point_d.append(p[i])
#     for n in nelson_coord:
#         nelson_d.append(n[i])
#     nelson_analysis = get_distance_error(nelson_d, coord_test)
#     with open('data_report{}.csv'.format(i), 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(
#             ['Actual_coord', 'Target_RSSI', 'Shift_Point1', 'Shift_Point2', 'Shift_toward_point', 'Shift',
#              'Accuracy', 'Pred_coord'])
#         for j in range(len(Y_random)):
#             row = [str(coord_test[j]), X_test2[j], point_d[j][0], point_d[j][1], point_d[j][2], i,
#                    str(nelson_analysis[j]), str(nelson_d[j])]
#             # row = [str(nelson_d[j]), str(coord_test[j]), str(Y_random[j]), str(nelson_analysis[j])]
#             csv_writer.writerow(row)
#     print(mean(nelson_analysis))

# MA_analysis = get_distance_error(module_A, random_targets)
# print(random_targets)
# print("*****************************")
# print(module_A)
# print("*****************************")
# print(MA_analysis)
equal_set = {('A', 'B', 'E'): ('B', 'A', 'E'), ('B', 'A', 'E'): ('A', 'B', 'E'),
             ('A', 'E', 'B'): ('E', 'A', 'B'), ('E', 'A', 'B'): ('A', 'E', 'B'),
             ('E', 'B', 'A'): ('B', 'E', 'A'), ('B', 'E', 'A'): ('E', 'B', 'A'),
             ('A', 'E', 'C'): ('E', 'A', 'C'), ('E', 'A', 'C'): ('A', 'E', 'C'),
             ('A', 'C', 'E'): ('C', 'A', 'E'), ('C', 'A', 'E'): ('A', 'C', 'E'),
             ('C', 'E', 'A'): ('E', 'C', 'A'), ('E', 'C', 'A'): ('C', 'E', 'A'),
             ('C', 'E', 'D'): ('E', 'C', 'D'), ('E', 'C', 'D'): ('C', 'E', 'D'),
             ('C', 'D', 'E'): ('D', 'C', 'E'), ('D', 'C', 'E'): ('C', 'D', 'E'),
             ('D', 'E', 'C'): ('E', 'D', 'C'), ('E', 'D', 'C'): ('D', 'E', 'C'),
             ('D', 'B', 'E'): ('B', 'D', 'E'), ('B', 'D', 'E'): ('D', 'B', 'E'),
             ('E', 'B', 'D'): ('B', 'E', 'D'), ('B', 'E', 'D'): ('E', 'B', 'D'),
             ('E', 'D', 'B'): ('D', 'E', 'B'), ('D', 'E', 'B'): ('E', 'D', 'B')}
letter_labels = ['C', 'A', 'E', 'D', 'B']
letter_dict = {5: 'A', 15: 'B', 1: 'C', 11: 'D', 8: 'E'}
letter_point = {'C': grid_points[24], 'A': grid_points[28], 'E': grid_points[31], 'D': grid_points[34],
                'B': grid_points[38]}
# letter_dict = {'4': 'A', '22': 'B', '1': 'C', '19': 'D', '31': 'E'}
# letter_dict = {'6': 'A', '24': 'B', '1': 'C', '19': 'D', '32': 'E'}
rf_model = RandomForestClassifier()
rf_model.fit(X_corner, y_corner)
rf_result = list(rf_model.predict_proba(X_train2))
best_rf = get_closet_points(rf_result, letter_labels)
rf_best = [b[-1] for b in best_rf]

# parameters = {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'C': [0.1, 1, 10, 100]}
# svc = SVC()
# svm_clf = GridSearchCV(svc, parameters)
# search_result = svm_clf.fit(X_corner, y_corner)
# print(search_result.best_params_)
# print(search_result.best_estimator_)

svm_model = SVC(decision_function_shape='ovo', probability=True, gamma=0.001, C=0.1)
svm_model.fit(X_corner, y_corner)
svm_result = list(svm_model.predict_proba(X_train2))
best_svm = get_closet_points(svm_result, letter_labels)
svm_best = [b[-1] for b in best_svm]

knn3_model = KNeighborsClassifier(n_neighbors=3)
knn3_model.fit(X_corner, y_corner)
knn3_result = list(knn3_model.predict_proba(X_train2))
best_knn3 = get_closet_points(knn3_result, letter_labels)
knn3_best = [b[-1] for b in best_knn3]

knn5_model = KNeighborsClassifier(n_neighbors=5)
knn5_model.fit(X_corner, y_corner)
knn5_result = list(knn5_model.predict_proba(X_train2))
best_knn5 = get_closet_points(knn5_result, letter_labels)
knn5_best = [b[-1] for b in best_knn5]

knn7_model = KNeighborsClassifier(n_neighbors=7)
knn7_model.fit(X_corner, y_corner)
knn7_result = list(knn7_model.predict_proba(X_train2))
best_knn7 = get_closet_points(knn7_result, letter_labels)
knn7_best = [b[-1] for b in best_knn7]

knn10_model = KNeighborsClassifier(n_neighbors=10)
knn10_model.fit(X_corner, y_corner)
knn10_result = list(knn10_model.predict_proba(X_train2))
best_knn10 = get_closet_points(knn10_result, letter_labels)
knn10_best = [b[-1] for b in best_knn10]

knn15_model = KNeighborsClassifier(n_neighbors=15)
knn15_model.fit(X_corner, y_corner)
knn15_result = list(knn15_model.predict_proba(X_train2))
best_knn15 = get_closet_points(knn15_result, letter_labels)
knn15_best = [b[-1] for b in best_knn15]

actual_result = find_closet(coord_train, letter_dict)
rf_accuracy = metrics.accuracy_score(actual_result, rf_best)
svm_accuracy = metrics.accuracy_score(actual_result, svm_best)
knn3_accuracy = metrics.accuracy_score(actual_result, knn3_best)
knn5_accuracy = metrics.accuracy_score(actual_result, knn5_best)
knn7_accuracy = metrics.accuracy_score(actual_result, knn7_best)
knn10_accuracy = metrics.accuracy_score(actual_result, knn10_best)
knn15_accuracy = metrics.accuracy_score(actual_result, knn15_best)

with open('data_compare.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    write_list = ['RSSI', 'actual_cp', 'RF_cp', 'SVM_cp',
                  '3NN_cp', '5NN_cp', '7NN_cp',
                  '10NN_cp', '15NN_cp']
    csv_writer.writerow(write_list)
    for i in range(len(actual_result)):
        csv_writer.writerow([X_train2[i], actual_result[i], rf_best[i], svm_best[i],
                             knn3_best[i], knn5_best[i], knn7_best[i],
                             knn10_best[i], knn15_best[i]])
    csv_writer.writerow(["Total", "", rf_accuracy, svm_accuracy, knn3_accuracy,
                         knn5_accuracy, knn7_accuracy, knn10_accuracy, knn15_accuracy])

rf_processes = list()
svm_processes = list()
knn5_processes = list()
# ONE POINT
# for i in range(len(best_rf)):
#     rf_shift, rf_list = calculateJSApproach(best_rf[i], coord_train[i], letter_point)
#     svm_shift, svm_list = calculateJSApproach(best_svm[i], coord_train[i], letter_point)
#     knn5_shift, knn5_list = calculateJSApproach(best_knn5[i], coord_train[i], letter_point)
#     rf_process = [best_rf[i][0], best_rf[i][1], best_rf[i][2], coord_train[i], X_train2[i],
#                   rf_shift, rf_list[rf_shift][1], rf_list[rf_shift][2]]
#     svm_process = [best_svm[i][0], best_svm[i][1], best_svm[i][2], coord_train[i], X_train2[i],
#                    svm_shift, svm_list[svm_shift][1], svm_list[svm_shift][2]]
#     knn5_process = [best_knn5[i][0], best_knn5[i][1], best_knn5[i][2], coord_train[i], X_train2[i],
#                     knn5_shift, knn5_list[knn5_shift][1], knn5_list[knn5_shift][2]]
#     for rf in rf_list:
#         rf_process += rf
#     for s in svm_list:
#         svm_process += s
#     for k in knn5_list:
#         knn5_process += k
#     rf_processes.append(rf_process)
#     svm_processes.append(svm_process)
#     knn5_processes.append(knn5_process)
# TWO POINT
for i in range(len(best_rf)):
    rf_shift, rf_list = calculateTwoShift(best_rf[i], coord_train[i], letter_point)
    svm_shift, svm_list = calculateTwoShift(best_svm[i], coord_train[i], letter_point)
    knn5_shift, knn5_list = calculateTwoShift(best_knn5[i], coord_train[i], letter_point)
    rf_process = [best_rf[i][0], best_rf[i][1], best_rf[i][2], coord_train[i], X_train2[i],
                  rf_shift, rf_list[rf_shift][0], rf_list[rf_shift][1]]
    svm_process = [best_svm[i][0], best_svm[i][1], best_svm[i][2], coord_train[i], X_train2[i],
                   svm_shift, svm_list[svm_shift][0], svm_list[svm_shift][1]]
    knn5_process = [best_knn5[i][0], best_knn5[i][1], best_knn5[i][2], coord_train[i], X_train2[i],
                    knn5_shift, knn5_list[knn5_shift][0], knn5_list[knn5_shift][1]]
    for key, value in rf_list.items():
        coord = value[-1]
        rf_process.append(key)
        rf_process.append(value[0])
        rf_process.append(coord)
    for key, value in svm_list.items():
        coord = value[-1]
        svm_process.append(key)
        svm_process.append(value[0])
        svm_process.append(coord)
    for key, value in knn5_list.items():
        coord = value[-1]
        knn5_process.append(key)
        knn5_process.append(value[0])
        knn5_process.append(coord)
    rf_processes.append(rf_process)
    svm_processes.append(svm_process)
    knn5_processes.append(knn5_process)

with open('rf_two_point.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    write_list = ['Shift_Point1', 'Shift_Point2', 'Toward_Point', 'Actual_Coord','Prefer_Coord'
                  'RSSI', 'Best_Shift', 'Best_accuracy', 'Best_coord']
    for i in range(100):
        write_list.append('Shift')
        write_list.append('Accuracy')
        write_list.append('Coord')
    csv_writer.writerow(write_list)
    for j in rf_processes:
        csv_writer.writerow(j)

with open('svm_two_point.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    write_list = ['Shift_Point1', 'Shift_Point2', 'Toward_Point', 'Actual_Coord', 'RSSI', 'Best_Shift',
                  'Best_accuracy', 'Best_coord']
    for i in range(100):
        write_list.append('Shift')
        write_list.append('Accuracy')
        write_list.append('Coord')
    csv_writer.writerow(write_list)
    for j in svm_processes:
        csv_writer.writerow(j)

with open('knn5_two_point.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    write_list = ['Shift_Point1', 'Shift_Point2', 'Toward_Point', 'Actual_Coord', 'RSSI', 'Best_Shift',
                  'Best_accuracy', 'Best_coord']
    for i in range(100):
        write_list.append('Shift')
        write_list.append('Accuracy')
        write_list.append('Coord')
    csv_writer.writerow(write_list)
    for j in knn5_processes:
        csv_writer.writerow(j)

optimum_data = dict()
learning_process = list()
for i in range(len(X_random)):
    d = X_train2[i]
    c = coord_train[i]

    sorted_points, normalized_rssi, dot_product = findNormalizedRSSI(d, c)
    optimum, accuracy_list = calculateTwoShift(dot_product, c)
    # optimum, accuracy_list = calculateJSApproach(dot_product, c)
    # label, optimum = calculateLinearApproach(sorted_points, c)
    normalized_value = list()
    process = [sorted_points[0], sorted_points[1], sorted_points[2], c, d,
               {dot_product[0][1]: dot_product[0][3]},
               {dot_product[1][1]: dot_product[1][3]},
               {dot_product[2][1]: dot_product[2][3]}, optimum,
               accuracy_list[optimum][1], accuracy_list[optimum][0]]
    for key, value in accuracy_list.items():
        coord = value[-1]
        process.append(key)
        process.append(coord)
        process.append(value[0])
    learning_process.append(process)
    if sorted_points in list(optimum_data.keys()):
        optimum_data[sorted_points].append([optimum, normalized_rssi])
    else:
        if equal_set[sorted_points] in list(optimum_data.keys()):
            optimum_data[equal_set[sorted_points]].append([optimum, normalized_rssi])
        else:
            optimum_data[sorted_points] = [[optimum, normalized_rssi]]

with open('data_training_rectangle.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    write_list = ['Shift_Point1', 'Shift_Point2', 'Toward_Point', 'Actual_Coord', 'RSSI', 'Normalized1', 'Normalized2',
                  'Normalized3', 'Best_Shift', 'Best_coord', 'Best_accuracy']
    for i in range(100):
        write_list.append('Shift')
        write_list.append('Coord')
        write_list.append('Accuracy')
    csv_writer.writerow(write_list)
    for r in learning_process:
        csv_writer.writerow(r)

svm_dict = dict()
for k, v in optimum_data.items():
    training_feature = [d[1] for d in v]
    training_class = [int("{}{}".format(d[0][0], d[0][1])) for d in v]
    if len(set(training_class)) == 1:
        svm_dict[k] = [training_class[0]]
        continue
    svm = SVC(decision_function_shape='ovo')
    svm.fit(training_feature, training_class)
    svm_dict[k] = svm

pred_result = list()
report_result = list()
for i in range(len(Y_random)):
    test_sample = Y_random[i]
    test_rssi = X_test2[i]
    actual_coord = coord_test[i]
    point_list, optimun_letter, average_rssi, dot_dict = determine_points(test_sample)
    # normalized_dict, sorted_points, normalized_rssi, dot_product = findKNNRSSI(d)
    # target_points = sorted([gp.num for gp in sorted_points])
    if optimun_letter in list(svm_dict.keys()):
        target_svm = svm_dict[optimun_letter]
    else:
        if equal_set[optimun_letter] in list(svm_dict.keys()):
            target_svm = svm_dict[equal_set[optimun_letter]]
        else:
            print("No Sample for this point:{}".format(actual_coord))
    if isinstance(target_svm, list):
        result = target_svm
    else:
        result = target_svm.predict([normalized_rssi])
    x = [int(a) for a in str(result[0])]
    if len(x) == 1:
        x = [0] + x
    coord_result = get_two_coord(point_list, x)
    # coord_result = get_coord(point_list, [result[0]])
    pred_result.append(coord_result[0])
    normalized_rssi_list = [{d[1]: d[-1]} for d in dot_dict[-3:]]
    dot_values = list()
    report_result.append(
        [optimun_letter[0], optimun_letter[1], optimun_letter[2],
         str(actual_coord), str(test_rssi), str(normalized_rssi_list[0]),
         str(normalized_rssi_list[1]), str(normalized_rssi_list[2]),
         str(x), str(coord_result[0])])

analysis = get_distance_error(pred_result, coord_test)
for i in range(len(analysis)):
    a = analysis[i]
    report_result[i].append(a)

print(pred_result)
print("*****************************")
print(analysis)
print("*****************************")
print(mean(analysis))

with open('data_report_new3.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        ['Shift_Point1', 'Shift_Point2', 'Toward_Point', 'Actual_Coord', 'RSSI', 'Normalized1', 'Normalized2',
         'Normalized3', 'Pred_Shift', 'Pred_coord', 'Pred_accuracy'])
    for r in report_result:
        csv_writer.writerow(r)

k = 8
k1 = 8
partition_list = svm_c(k, grid_points)
svmc_pattern = list()
for p in partition_list:
    svmc_pattern.append([z.num for z in p])

train_mode1 = 'svm'
train_mode2 = 'knn'
svmc_center = list()
for i in range(len(svmc_pattern)):
    svmc_group = svmc_pattern[i]
    svmc_center.append(find_center([get_gp_num(grid_points, gp) for gp in svmc_group]))

compare_report = get_result(train_mode1, train_mode2, svmc_pattern, k, 'SVM-C')

df = pd.DataFrame(center_inputs)
tf = pd.DataFrame(random_inputs)

kmeans = KMeans(n_clusters=k1, random_state=42).fit(X=df)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = df.index.values
cluster_map['cluster'] = kmeans.labels_

cluster_dict = dict()
for label in range(k1):
    cluster_data = cluster_map[cluster_map.cluster == label]
    cluster_list = list()
    for i, row in cluster_data.iterrows():
        index = row.data_index
        feature = center_inputs[index]
        answer = center_targets[index]
        cluster_list.append([answer, feature])
    cluster_dict[label] = cluster_list
k_offline_Y = list()
k_offline_X = list()
gp_list = list()
for key, value in cluster_dict.items():
    gp_dict = dict()
    for v in value:
        gp = v[0]
        k_offline_X.append(v[1])
        k_offline_Y.append(key)
        if gp in list(gp_dict.keys()):
            gp_dict[gp] += 1
        else:
            gp_dict[gp] = 1
    for i in range(len(gp_list)):
        item = gp_list[i]
        item_list = set(item.keys())
        duplicate_list = list(item_list.intersection(list(gp_dict.keys())))
        while len(duplicate_list) != 0:
            duplicate_value = duplicate_list[0]
            if gp_dict[duplicate_value] > item[duplicate_value]:
                gp_list[i].pop(duplicate_value)
            else:
                gp_dict.pop(duplicate_value)
            duplicate_list.pop(0)
    gp_list.append(gp_dict)

k_partition = list()
center_list = list()
for i in range(len(gp_list)):
    k_group = gp_list[i]
    k_partition.append(list(k_group.keys()))
    center_list.append(find_center([get_gp_num(grid_points, gp) for gp in list(k_group.keys())]))

k_report = get_result(train_mode1, train_mode2, k_partition, k1, 'k-means')
print("***********************")
# actual_partition = list()
# for x in range(len(random_inputs)):
#     actual_class = random_targets[x]
#     for index, p in enumerate(k_partition):
#         if actual_class in p:
#             actual_partition.append(index)
# kmeans_result = list(kmeans.predict(random_inputs))
# kmeans_corrd = [center_list[k] for k in kmeans_result]
# Kmeans_report = get_distance_error(kmeans_corrd, random_coords)
# print('\n')
# print("K-means testing Accuracy: {}".format(mean(Kmeans_report)))
# print("K-means classification Accuracy: {}".format(metrics.accuracy_score(actual_partition, kmeans_result)))

partition_pattern = space_partition(grid_points)
# hsvmc_pattern = find_pattern(k, partition_pattern)
# h_offline_Y = list()
# h_offline_X = list()
# for j in range(k):
#     gp_nums = hsvmc_pattern[j]
#     for i in range(len(offline_X)):
#         offline_row = center_inputs[i]
#         offline_class = center_targets[i]
#         if offline_class in gp_nums:
#             h_offline_Y.append(j)
#             h_offline_X.append(offline_row)
# h_center = list()
# for i in range(len(svmc_pattern)):
#     h_group = svmc_pattern[i]
#     h_center.append(find_center([get_gp_num(grid_points, gp) for gp in h_group]))
#
# h_report = get_result(train_mode, h_offline_X, h_offline_Y, hsvmc_pattern, h_center, 'HSVM-C')

HSVM_report = tree_build(train_mode1=train_mode1, train_mode2=train_mode2, pattern=partition_pattern,
                         gp_pattern=gp_pattern)
print("***********************")
train_mode3 = 'knn'
single_model = train_model(train_mode3, center_inputs, center_targets, k=5)
test_result = list()
test_class = list()
start_knn = time()
for n in range(len(random_inputs)):
    ap_test_feature = random_inputs[n]
    single_result = single_model.test_one(train_mode3, [ap_test_feature])
    test_class.append(single_result[0])
    test_result.append(get_gp_num(grid_points, single_result[0]).point)
end_knn = time()
print(end_knn - start_knn)
single_report = get_distance_error(test_result, random_coords)
print("{} result is {}".format(train_mode3, mean(single_report)))
print("{} result is {}".format(train_mode3, metrics.accuracy_score(random_targets, test_class)))

# stat1, p1 = mannwhitneyu(HSVM_report, k_report)
# print('HSVM-C vs KmeansStatistics=%.3f, p=%.3f' % (stat1, p1))
# # interpret
# alpha = 0.05
# if p1 > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')
#
# stat2, p2 = mannwhitneyu(HSVM_report, compare_report)
# print('HSVM-C vs SVM-C Statistics=%.3f, p=%.3f' % (stat2, p2))
# if p2 > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')
#
# stat3, p3 = mannwhitneyu(compare_report, k_report)
# print('SVM-C vs Kmeans Statistics=%.3f, p=%.3f' % (stat3, p3))
# if p3 > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')

# stat, p = normaltest(HSVM_report)
# print('HSVM-C Statistics=%.3f, p=%.3f' % (stat, p))
# # interpret
# alpha = 0.05
# if p > alpha:
#     print('Sample looks Gaussian (fail to reject H0)')
# else:
#     print('Sample does not look Gaussian (reject H0)')
#
# stat1, p1 = normaltest(compare_report)
# print('SVM-C Statistics=%.3f, p=%.3f' % (stat1, p1))
# if p1 > alpha:
#     print('Sample looks Gaussian (fail to reject H0)')
# else:
#     print('Sample does not look Gaussian (reject H0)')
#
# stat2, p2 = normaltest(k_report)
# print('K-means Statistics=%.3f, p=%.3f' % (stat2, p2))
# if p2 > alpha:
#     print('Sample looks Gaussian (fail to reject H0)')
# else:
#     print('Sample does not look Gaussian (reject H0)')
