from collections import Callable
import math

from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from typing import Tuple, List, Dict
import numpy as np


def get_calculation_function(calculation_mode: str) -> Callable:
    if calculation_mode == "NNv4":
        return get_NNv4_RSSI
    elif calculation_mode == "kNNv1":
        return get_KNNv1
    elif calculation_mode == "kNNv2":
        return get_KNNv2
    elif calculation_mode == "kNNv3":
        return get_KNNv3
    else:
        raise Exception("The Calculation Mode: {} is invalid.".format(calculation_mode))


def get_NNv4_RSSI(centroid_points: List[Centroid], rssis: Dict[AccessPoint, int] = None,
                  list_rssis: Dict[Centroid, Dict[AccessPoint, int]] = None) -> Tuple[float, float]:
    colest_centroid = centroid_points[0]
    smallest_distance = 0
    if rssis is not None and list_rssis is None:
        for cp in centroid_points:
            point = cp.Center
            distances = 0
            for ap, rssi in rssis.items():
                value = point.get_rssis(ap)
                distance = math.pow((rssi - value), 2)
                distances += distance
            distances = math.sqrt(distances)
            # print("distance betweem centroid " + point.id + " and target is " + str(distances))
            if cp == centroid_points[0]:
                smallest_distance = distances
            if distances < smallest_distance:
                smallest_distance = distances
                colest_centroid = cp
    if list_rssis is not None and rssis is None:
        for cp, rssis in list_rssis.items():
            point = cp.Center
            distances = 0
            for ap, rssi in rssis.items():
                distance = math.pow((rssi - point.get_rssis(ap)), 2)
                distances += distance
            distances = math.sqrt(distances)
            # print("distance betweem centroid " + point.id + " and target is " + str(distances))
            if cp == centroid_points[0]:
                smallest_distance = distances
            if distances < smallest_distance:
                smallest_distance = distances
                colest_centroid = cp
    # return colest_centroid
    return colest_centroid.point


def get_KNNv1(centroid_points: List[Centroid], rssis: Dict[AccessPoint, int],
              grid_points_only: List[GridPoint] = None) -> Tuple[float, float]:
    """
    :param grid_points_only:
    :param centroid_points: List of Centroid Points to search against.
    :param rssis: Dict of Access Point: RSSI values.
    :return: A tuple of X, Y denoting the position related to the passed RSSIs.
    """

    for cp in centroid_points:
        if grid_points_only is None:
            grid_points_only = cp.CornerPoints
        else:
            for cnp in cp.CornerPoints:
                if cnp in grid_points_only:
                    pass
                else:
                    grid_points_only.append(cnp)

    for gp in grid_points_only:
        distances = 0
        for ap, rssi in rssis.items():
            distance = math.pow((rssi - gp.get_rssis(ap)), 2)
            distances += distance
        try:
            distances = math.sqrt(distances)
            gp.distance = distances
            # gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print("Oop!")

        # Sort the grid points by distance:
        ####  ref: closest_cp.sort_corner_points_by_distance()

        # Selection sort
    for i in range(len(grid_points_only)):
        min_index = i
        for j in range(i + 1, len(grid_points_only)):
            if grid_points_only[min_index] > grid_points_only[j]:
                min_index = j

        grid_points_only[i], grid_points_only[min_index] = grid_points_only[min_index], grid_points_only[i]

        #### TODO: could consider use other sorting algorithms

        # Create a new central point of the K (current K=3) proximate Gird-Points:
    x = round(((grid_points_only[0].x + grid_points_only[1].x + grid_points_only[2].x) / 3), 2)
    y = round(((grid_points_only[0].y + grid_points_only[1].y + grid_points_only[2].y) / 3), 2)
    print("{},{}".format(x, y))
    return x, y


def get_KNNv2(centroid_points: List[Centroid], rssis: Dict[AccessPoint, int] = None,
              list_rssis: Dict[Centroid, Dict[AccessPoint, int]] = None) -> Tuple[float, float]:
    """
    :param list_rssis:
    :param centroid_points: List of Centroid Points to search against.
    :param rssis: Dict of Access Point: RSSI values.
    :return: A tuple of X, Y denoting the position related to the passed RSSIs.
    """

    # print(rssis)
    if rssis is not None and list_rssis is None:
        # 1. For every Centroid, calculate the distance:
        for cp in centroid_points:
            point = cp.Center
            point.distance = 0
            distances = 0
            for ap, rssi in rssis.items():
                distance = math.pow((rssi - point.get_rssis(ap)), 2)
                distances += distance
            distances = math.sqrt(distances)
            point.distance = distances

    if list_rssis is not None and rssis is None:
        for cp, rssis in list_rssis.items():
            point = cp.Center
            point.distances = 0
            distances = 0
            for ap, rssi in rssis.items():
                distance = math.pow((rssi - point.get_rssis(ap)), 2)
                distances += distance
            distances = math.sqrt(distances)
            point.distance = distances
    # 2. Sort the centroid points by distance:
    ####  ref: sort_corner_points_by_distance()

    # Selection sort
    for i in range(len(centroid_points)):
        min_index = i
        for j in range(i + 1, len(centroid_points)):
            if centroid_points[min_index].Center.distance > centroid_points[j].Center.distance:
                min_index = j

        centroid_points[i], centroid_points[min_index] = centroid_points[min_index], centroid_points[i]

    #### TODO: could consider use other sorting algorithms

    # 3. Create a new centroid point of the K (current K=3) proximate Centroid-Points:
    x = (centroid_points[0].x + centroid_points[1].x + centroid_points[2].x) / 3
    y = (centroid_points[0].y + centroid_points[1].y + centroid_points[2].y) / 3

    return x, y


def get_KNNv3(grid_points: List[GridPoint], rssis: Dict[AccessPoint, int]) -> Tuple[float, float]:
    """
    :param grid_points: List of Grid Points to search against.
    :param rssis: Dict of Access Point: RSSI values.
    :return: A tuple of X, Y denoting the position related to the passed RSSIs.
    """

    # For every grid point, calculate the distance:
    for gp in grid_points:
        # point = gp  # possible issue
        # point.distance = 0
        distances = 0
        for ap, rssi in rssis.items():
            distance = math.pow((rssi - gp.get_rssis(ap)), 2)
            distances += distance
        try:
            distances = math.sqrt(distances)
            gp.distance = distances
            # gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print("Oop!")

    # Sort the grid points by distance:
    ####  ref: closest_cp.sort_corner_points_by_distance()

    # Selection sort
    for i in range(len(grid_points)):
        min_index = i
        for j in range(i + 1, len(grid_points)):
            if grid_points[min_index] > grid_points[j]:
                min_index = j

        grid_points[i], grid_points[min_index] = grid_points[min_index], grid_points[i]

    #### TODO: could consider use other sorting algorithms

    # Create a new central point of the K (current K=3) proximate Gird-Points:
    x = round(((grid_points[0].x + grid_points[1].x + grid_points[2].x) / 3), 2)
    y = round(((grid_points[0].y + grid_points[1].y + grid_points[2].y) / 3), 2)

    return x, y
