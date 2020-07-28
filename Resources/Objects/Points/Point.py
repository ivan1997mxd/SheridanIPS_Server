from abc import ABC, abstractmethod
from typing import List


# region Class
class Point(ABC):

    __points = list()   # type: List[Point]

    # region Constructor
    def __init__(self, id: str, num: int, x: float, y: float, z: float = 0):
        self.__ID = id
        self.__Num = num
        self.__X = x
        self.__Y = y
        self.__Z = z

        Point.__points.append(self)
    # endregion

    # region Properties
    @property
    def id(self) -> str:
        return str(self.__ID)

    @property
    def num(self) -> int:
        return self.__Num

    @property
    def point(self) -> (float, float):
        return self.__X, self.__Y

    @property
    def x(self) -> float:
        return self.__X

    @property
    def y(self) -> float:
        return self.__Y

    @property
    def z(self) -> float:
        return self.__Z
    # endregion

    def __repr__(self) -> str:
        return str(self.num)

    # region Static Method
    @staticmethod
    def approx_equal(distance1: float, distance2: float) -> bool:
        return abs(distance1 - distance2) < 0.25

    @staticmethod
    def reset_points():
        for point in Point.__points:
            point._reset()

    @classmethod
    @abstractmethod
    def create_point_list(cls, file_path: str, *args, **kwargs) -> list:
        pass

    @abstractmethod
    def _reset(self):
        pass
    # endregion
# endregion
