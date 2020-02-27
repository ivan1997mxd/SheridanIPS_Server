from Resources.Objects.Points.Point import Point
from Resources.Objects.Points.AccessPoint import AccessPoint
from typing import Dict, List
import csv


class GridPoint(Point):

    def __init__(self, id: str, num: int, ple_vals: Dict[AccessPoint, float], x: float, y: float, z: int = 0):
        super(GridPoint, self).__init__(id=id, num=num, x=x, y=y, z=z)
        self.__pleValues = ple_vals     # type: Dict[AccessPoint, float]
        self.__distance: float = -1

    # region Properties
    @property
    def distance(self) -> float:
        return self.__distance
    # endregion

    # region Setters
    @distance.setter
    def distance(self, distance: float) -> None:
        self.__distance = distance

    # def set_average_rssi(self, access_point: AccessPoint, rssis: [int]) -> None:
    #     assert len(rssis) > 0, "RSSIs can not be empty."
    #     self.__average_rssi[access_point] = sum(rssis) / len(rssis)
    # endrgion

    # region Getters
    def get_pleValue(self, access_point: AccessPoint) -> float:
        return self.__pleValues[access_point]

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

                ple_vals = dict()
                for i in range(len(line[3:]) - 1):
                    ap_bssid = line[3 + i]
                    ple = line[4 + i]

                    for ap in access_points:
                        if ap.id == ap_bssid:
                            ple_vals[ap] = float(ple)

                points.append(GridPoint(id=str(point_num), num=int(point_num), x=x_val, y=y_val, ple_vals=ple_vals))

        return points
    # endregion

