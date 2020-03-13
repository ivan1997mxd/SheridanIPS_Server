import csv
from typing import List, Tuple


class Zone:
    def __init__(self, num: int, tl_x: float, tl_y: float, br_x: float, br_y: float):
        self.__num: int = num
        self.__top_left_x: float = tl_x
        self.__top_left_y: float = tl_y
        self.__bottom_right_x: float = br_x
        self.__bottom_right_y: float = br_y

    # region Properties
    @property
    def num(self) -> int:
        return self.__num

    @property
    def top_left(self) -> Tuple[float, float]:
        return self.__top_left_x, self.__top_left_y

    @property
    def bottom_right(self) -> Tuple[float, float]:
        return self.__bottom_right_x, self.__bottom_right_y
    # endregion

    def __str__(self) -> str:
        if self.__num == 6:
            return "Unknown Actual Zone"
        return "Zone " + str(self.__num)

    def contains(self, point: Tuple[float, float]) -> bool:
        x = point[0]
        y = point[1]
        if self.__top_left_x <= x <= self.__bottom_right_x:
            if self.__top_left_y <= y <= self.__bottom_right_y:
                return True
        return False


def get_all_zones(file_path: str) -> List[Zone]:
    zones = list()  # type: List[Zone]

    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")

        for row_num, line in enumerate(readCSV):
            point_num = int(line[0])
            tl_x_val = float(line[1])
            tl_y_val = float(line[2])
            br_x_val = float(line[3])
            br_y_val = float(line[4])

            zones.append(Zone(point_num, tl_x_val, tl_y_val, br_x_val, br_y_val))

    return zones


def get_zone(zones: List[Zone], co_ordinate: Tuple[float, float]) -> Zone:
    for zone in zones:
        if zone.contains(co_ordinate):
            return zone
    raise Exception("Zone not found. Co-ordinates: " + str(co_ordinate))
