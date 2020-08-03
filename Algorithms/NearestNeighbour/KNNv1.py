from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint import GridPoint
from typing import Tuple, List, Dict
import numpy as np
import warnings

# KNNv1.py - by Margaret Xie - On May 11, 2020, following NNv4.py

def __euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))


def get_KNNv1(grid_points: List[GridPoint], rssis: Dict[AccessPoint, int]) -> Tuple[float, float]:
    """
    :param grid_points: List of Grid Points to search against.
    :param rssis: Dict of Access Point: RSSI values.
    :return: A tuple of X, Y denoting the position related to the passed RSSIs.
    """
    
    print(rssis) # comment it later

    # For every grid point, calculate the distance:
    for gp in grid_points:
        # point = gp  # possible issue
        # point.distance = 0

        for ap, rssi in rssis.items():
            euclidean_distance = __euclidean_distance(ap.point, gp.point)
            ple_distance = ap.ple_distance(rssi, point.get_pleValue(ap))

            gp.distance += np.power(euclidean_distance - ple_distance, 2)
            #if gp.distance < 0: print("Oop!!")
        try:
            gp.distance = np.sqrt(gp.distance)
        except RuntimeWarning:
            print ("Oop!")


    # Sort the grid points by distance:
    ####  ref: closest_cp.sort_corner_points_by_distance()

    # Selection sort
    for i in range(len(grid_points)):
        min_index = i
        for j in range(i+1, len(grid_points)):
            if grid_points[min_index] > grid_points[j]:
                min_index = j

        grid_points[i], grid_points[min_index] = grid_points[min_index], grid_points[i]

    #### TODO: could consider use other sorting algorithms


    # Create a new central point of the K (current K=3) proximate Gird-Points:
    x = (grid_points[0].x + grid_points[1].x + grid_points[2].x) / 3
    y = (grid_points[0].y + grid_points[1].y + grid_points[2].y) / 3

    return x, y
