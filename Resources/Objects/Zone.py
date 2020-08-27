import csv
from typing import List, Tuple
from Resources.Objects.Points.Centroid import Centroid


# class Zone:
#     def __init__(self, num: int, tl_x: float, tl_y: float, br_x: float, br_y: float):
#         self.__num: int = num
#         self.__top_left_x: float = tl_x
#         self.__top_left_y: float = tl_y
#         self.__bottom_right_x: float = br_x
#         self.__bottom_right_y: float = br_y
#
#     # region Properties
#     @property
#     def num(self) -> int:
#         return self.__num
#
#     @property
#     def top_left(self) -> Tuple[float, float]:
#         return self.__top_left_x, self.__top_left_y
#
#     @property
#     def bottom_right(self) -> Tuple[float, float]:
#         return self.__bottom_right_x, self.__bottom_right_y
#
#     # endregion
#
#     def __str__(self) -> str:
#         if self.__num == 6:
#             return "Unknown Actual Zone"
#         return "Zone " + str(self.__num)
#
#     def contains(self, point: Tuple[float, float]) -> bool:
#         x = point[0]
#         y = point[1]
#         if self.__top_left_x <= x <= self.__bottom_right_x:
#             if self.__top_left_y <= y <= self.__bottom_right_y:
#                 return True
#         return False


class Zone:
    def __init__(self, zone_num: str, room_id: str, points: Centroid):
        self.zone_num: str = zone_num
        self.room_id: str = room_id
        self.points: Centroid = points

    @property
    def num(self) -> int:
        num = int(self.zone_num[-1])
        return num

    @property
    def center(self) -> tuple:
        cood = self.points.point
        return cood

    def __str__(self) -> str:
        return "Zone " + str(self.num)

    def __lt__(self, other):
        return True if self.num < other.num else False

    def __le__(self, other):
        return True if self.num <= other.num else False

    def __gt__(self, other):
        return True if self.num > other.num else False

    def __ge__(self, other):
        return True if self.num >= other.num else False

    @classmethod
    def create_point_list(cls, zone_list: list, gp_list: list):
        zones = list()
        for z in zone_list:
            zone_num = z["zone_Num"]
            room_id = z['room_id']
            points = Centroid.create_point(z['points'], gp_list)
            zones.append(Zone(zone_num=zone_num, room_id=room_id, points=points))
        return zones

    def home_contains(self, point: Tuple[float, float]) -> bool:
        x = point[0]
        y = point[1]
        if self.points.BottomRight.x <= x <= self.points.TopLeft.x:
            if self.points.TopLeft.y <= y <= self.points.BottomRight.y:
                return True
        return False

    def contains(self, point: Tuple[float, float]) -> bool:
        x = point[0]
        y = point[1]
        if self.points.TopLeft.x <= x <= self.points.BottomRight.x:
            if self.points.TopLeft.y <= y <= self.points.BottomRight.y:
                return True
        return False

    def get_closest(self, point: Tuple[float, float]) -> bool:
        x = point[0]
        y = point[1]
        if self.num in [1, 2, 3]:
            if self.points.BottomRight.x <= x <= self.points.TopLeft.x + 2.60:
                if self.points.TopLeft.y <= y <= self.points.BottomRight.y:
                    return True
            return False
        if self.num in [5, 6]:
            if self.points.BottomRight.x <= x <= self.points.TopLeft.x:
                if self.points.TopLeft.y - 1.0 <= y <= self.points.BottomRight.y:
                    return True
            return False


# def get_all_zones(file_path: str) -> List[Zone]:
#     zones = list()  # type: List[Zone]
#
#     with open(file_path) as csvfile:
#         readCSV = csv.reader(csvfile, delimiter=",")
#
#         for row_num, line in enumerate(readCSV):
#             point_num = int(line[0])
#             tl_x_val = float(line[1])
#             tl_y_val = float(line[2])
#             br_x_val = float(line[3])
#             br_y_val = float(line[4])
#
#             zones.append(
#                 Zone(point_num, tl_x_val, tl_y_val, br_x_val, br_y_val))
#
#     return zones


def get_zone(zones: List[Zone], co_ordinate: Tuple[float, float]) -> Zone:
    for zone in zones:
        if zone.room_id == "Bedroom" or zone.room_id == "Hallway":
            if zone.home_contains(co_ordinate):
                return zone
            else:
                if zone.get_closest(co_ordinate):
                    return zone
        else:
            if zone.contains(co_ordinate):
                return zone
    raise Exception("Zone not found. Co-ordinates: " + str(co_ordinate))


def get_all_zones_db(zone_list: list) -> List[Zone]:
    zones = list()  # type: List[Zone]

    for zone in zone_list:
        point_num = int(zone['point_num'])
        tl_x_val = float(zone['tl_x_val'])
        tl_y_val = float(zone['tl_y_val'])
        br_x_val = float(zone['br_x_val'])
        br_y_val = float(zone['br_y_val'])

        zones.append(
            Zone(point_num, tl_x_val, tl_y_val, br_x_val, br_y_val))

    return zones
