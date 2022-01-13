import math

from Resources.Objects.Points.Point import Point
from Resources.Objects.Points.AccessPoint import AccessPoint
from typing import Dict, List
import csv


class GridPoint(Point):

    def __init__(self, id: str, num: int, rssis: Dict[AccessPoint, int], x: float, y: float, z: float = 0.0,
                 margins: dict = None):
        super(GridPoint, self).__init__(id=id, x=x, y=y, z=z, num=num)
        self.__rssis = rssis  # type: Dict[AccessPoint, int]
        self.__distance: float = -1
        self.__margins = margins

    # region Properties
    @property
    def distance(self) -> float:
        return self.__distance

    @property
    def margin(self) -> dict:
        return self.__margins

    @property
    def rssi(self) -> list:
        return list(self.__rssis.values())

    # endregion
    def __str__(self) -> str:
        return "GP" + str(self.num)

    # region Setters
    @distance.setter
    def distance(self, distance: float) -> None:
        self.__distance = distance

    # def set_average_rssi(self, access_point: AccessPoint, rssis: [int]) -> None:
    #     assert len(rssis) > 0, "RSSIs can not be empty."
    #     self.__average_rssi[access_point] = sum(rssis) / len(rssis)
    # endrgion

    # region Getters
    def get_rssis(self, access_point: AccessPoint) -> int:
        return self.__rssis[access_point]

    def get_str_rssis(self, ap_str: str) -> int:
        for ap in self.__rssis.keys():
            if ap_str == ap.id:
                return self.__rssis[ap]

    # def get_average_rssi(self, access_point: AccessPoint) -> float:
    #     return self.__average_rssi[access_point]
    # endregion

    # region Comparison Operators
    def __lt__(self, other):
        return True if self.distance < other.distance else False

    def __le__(self, other):
        return True if self.distance <= other.distance else False

    def __gt__(self, other):
        return True if self.distance > other.distance else False

    def __ge__(self, other):
        return True if self.distance >= other.distance else False

    def __eq__(self, other):
        return True if GridPoint.approx_equal(self.distance, other.distance) else False

    # endregion

    # region Method Overrides
    def _reset(self):
        self.__distance = -1

    @classmethod
    def create_point_list(cls, file_path: str, *args, **kwargs) -> list:
        assert len(args) == 0, "Grid Points are unable to accept variable arguments."
        assert "access_points" in kwargs.keys(), "You must pass a list of Access Points."

        access_points = kwargs["access_points"]
        points = list()

        with open(file_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")

            for row_num, line in enumerate(readCSV):

                if row_num == 0:
                    continue

                point_num = int(line[0])
                x_val = float(line[1])
                y_val = float(line[2])

                rssis = dict()
                for i in range(len(line[3:]) - 1):
                    ap_bssid = line[3 + i]
                    rssi = line[4 + i]

                    for ap in access_points:
                        if ap.id == ap_bssid:
                            rssis[ap] = int(rssi)

                points.append(GridPoint(id=str(point_num), num=point_num, x=x_val, y=y_val, rssis=rssis))

        return points

    @classmethod
    def create_point_list_db(cls, gp_list: list, ap_list: list) -> list:
        points = list()
        for gp in gp_list:
            rssis = dict()
            point_id = str(gp['gpnum'])
            point_num = int(point_id[6:])
            x_val = float(gp['x'])
            y_val = float(gp['y'])
            z_val = float(gp['z'])
            values = gp['values']
            for key, value in values.items():
                for ap in ap_list:
                    if key == ap.id:
                        rssis[ap] = int(value)
            points.append(GridPoint(id=point_id, num=point_num, x=x_val, y=y_val, z=z_val, rssis=rssis))

        return points

    @classmethod
    def create_point_list_db_new(cls, gp_list: list, ap_list: list) -> list:
        points = list()
        for gp in gp_list:
            rssis = dict()
            point_id = str(gp['gpnum'])
            point_num = int(point_id)
            x_val = float(gp['x'])
            y_val = float(gp['y'])
            z_val = float(gp['z'])
            distances = gp['distance']
            values = gp['values']
            for key, value in values.items():
                for ap in ap_list:
                    if key == ap.id:
                        rssis[ap] = int(value)
            points.append(
                GridPoint(id=point_id, num=point_num, x=x_val, y=y_val, z=z_val, rssis=rssis, margins=distances))

        return points


def get_gp_num(gps: List[GridPoint], gp_num: int) -> GridPoint:
    for gp in gps:
        if gp.num == gp_num:
            return gp
    raise Exception("GP not found. num: " + str(gp_num))


def find_cloest_p(gps: list, gp_coord: list) -> GridPoint:
    cloest_gp = 0
    cloest_distance = 100
    for i in range(len(gps)):
        actual_coord = gps[i].point
        pred_distance = math.sqrt(
            math.pow((gp_coord[0] - actual_coord[0]), 2) + math.pow((gp_coord[1] - actual_coord[1]), 2))
        if pred_distance < cloest_distance:
            cloest_gp = i + 1
            cloest_distance = pred_distance

    return get_gp_num(gps, cloest_gp)
