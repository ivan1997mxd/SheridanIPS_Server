import math

from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from typing import Tuple, List, Dict
import numpy as np
import warnings


def __euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))


def get_NNv4_RSSI(centroid_points: List[Centroid], rssis: Dict[AccessPoint, int]) -> Tuple[float, float]:
    cloest_centroid = centroid_points[0]
    smallest_distance = 0
    for cp in centroid_points:
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
            cloest_centroid = cp

    return cloest_centroid.point


def get_NNv4(centroid_points: List[Centroid], rssis: Dict[AccessPoint, int]) -> Tuple[float, float]:
    """
    :param centroid_points: List of Centroid Points to search against.
    :param rssis: Dict of Access Point: RSSI values.
    :return: A tuple of X, Y denoting the position related to the passed RSSIs.
    """

    # print(rssis)

    # For every Centroid, calculate the distance:
    for cp in centroid_points:
        point = cp.Center
        point.distance = 0

        for ap, rssi in rssis.items():
            euclidean_distance = __euclidean_distance(ap.point, point.point)
            ple_distance = ap.ple_distance(rssi, point.get_pleValue(ap))

            point.distance += np.power(euclidean_distance - ple_distance, 2)

        point.distance = np.sqrt(point.distance)

    # Get the closest Centroid:
    closest_cp = centroid_points[0]
    for cp in centroid_points:
        if cp.Center.distance < closest_cp.Center.distance:
            closest_cp = cp

    # For every Grid Point in the Centroid, calculate the distance:
    for gp in closest_cp.CornerPoints:
        # warnings.simplefilter("error")
        # if gp.num == 13: print("GP == 13")
        for ap, rssi in rssis.items():
            euclidean_distance = __euclidean_distance(ap.point, gp.point)
            ple_distance = ap.ple_distance(rssi, gp.get_pleValue(ap))

            gp.distance += np.power(euclidean_distance - ple_distance, 2)
            # if gp.distance < 0: print("Oop!!")
        try:
            gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print("Oop!")

    # Sort the grid points by distance:
    closest_cp.sort_corner_points_by_distance()

    # Get the number of approximately equidistant points:
    # TODO: Maybe make this a class method in the Centroid Class?
    num_of_equidistant_points = 0
    for i in range(0, len(closest_cp.CornerPoints)):
        for j in range(i + 1, len(closest_cp.CornerPoints)):
            if GridPoint.approx_equal(closest_cp.CornerPoints[i].distance, closest_cp.CornerPoints[j].distance):
                num_of_equidistant_points += 1

    # Case 1: At least 3 of the 4 points are equidistant.
    if num_of_equidistant_points >= 2:
        return closest_cp.Center.point

    # Case 2: The closest 2 points are equidistant.
    if num_of_equidistant_points == 1:
        # Return their midpoint.
        x = (closest_cp.CornerPoints[0].x + closest_cp.CornerPoints[1].x) / 2
        y = (closest_cp.CornerPoints[0].y + closest_cp.CornerPoints[1].y) / 2
        return x, y

    # Case 3: Nothing is equidistant
    # Create a new central point, between the Centroid and the two closest Corner Points:
    x = (closest_cp.Center.x + closest_cp.CornerPoints[0].x + closest_cp.CornerPoints[1].x) / 3
    y = (closest_cp.Center.y + closest_cp.CornerPoints[0].y + closest_cp.CornerPoints[1].y) / 3

    # Return the midpoint of the centroid, closest point, and the new central point just created.
    midx = (closest_cp.Center.x + closest_cp.CornerPoints[0].x + x) / 3
    midy = (closest_cp.Center.y + closest_cp.CornerPoints[0].y + y) / 3
    return midx, midy
