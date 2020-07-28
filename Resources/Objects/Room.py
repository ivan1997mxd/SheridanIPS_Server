from abc import ABC
from typing import *
from Resources.Objects import Zone
from Resources.Objects.Points import AccessPoint


class Room(ABC):
    def __init__(self, num: str, floor: int, building: str, access_points: list, zones: list):
        # super(Room, self).__init__(num=num, zones=zones, floor=floor, building=building, access_points=access_points)
        self.__num = num                        # type: str
        self.__zones = zones                    # type: list
        self.__floor = floor                    # type: int
        self.__building = building              # type: str
        self.__access_points = access_points    # type: list

    # region Properties
    @property
    def num(self) -> str:
        return self.__num

    @property
    def floor(self) -> int:
        return self.__floor

    @property
    def building(self) -> str:
        return self.__building

    @property
    def access_points(self) -> list:
        return self.__access_points

    @property
    def zones(self) -> list:
        return self.__zones
