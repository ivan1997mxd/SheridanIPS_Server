from Resources.Objects.Points.AccessPoint import AccessPoint
from typing import List, Dict, Tuple, Union
from Resources.Objects.Zone import Zone
from abc import ABC
import csv


class Matrix(ABC):

    def __init__(self, access_points: List[AccessPoint], zones: List[Zone], size: int = 5):
        assert size > 0, "The matrix must have a side length greater than 0."
        assert len(access_points) > 1, "The matrices must have at least 2 Access Points."

        self.__size = size                      # type: int
        self.__access_points = access_points    # type: List[AccessPoint]
        self.__zones = zones                    # type: List[Zone]
        self.__csv_list = None                  # type: Union[None, List[List[str]]]
        self.__matrix_sum = -1                  # type: float
        self.__id = None                        # type: Union[None, str]
        self.__Vector_Dict = dict()             # type: Dict[Zone, Dict[Zone, float]]
        for measured_zone in zones:
            self.__Vector_Dict[measured_zone] = {actual_zone: 0 for actual_zone in zones}

    # region Properties
    @property
    def size(self) -> int:
        return self.__size

    @property
    def access_points(self) -> List[AccessPoint]:
        return self.__access_points

    @property
    def zones(self) -> List[Zone]:
        return self.__zones

    @property
    def matrix_sum(self) -> float:
        if self.__matrix_sum != -1:
            return self.__matrix_sum

        self.__matrix_sum = 0
        for vector in self.__Vector_Dict.values():
            self.__matrix_sum += sum(vector.values())
        return self.__matrix_sum

    @property
    def id(self) -> str:
        if self.__id is not None:
            return self.__id

        Str = ""
        for ap in self.__access_points:
            Str += str(ap.num)
        self.__id = Str
        return self.__id

    @property
    def measured_zones_and_vectors(self) -> Tuple[Zone, Dict[Zone, List[float]]]:
        for measured_zone, vector in self.__Vector_Dict.items():
            yield measured_zone, vector

    @property
    def vectors(self) -> Tuple[Zone, Dict[Zone, float]]:
        for measured_zone, vector in self.__Vector_Dict.items():
            yield measured_zone, vector

    @property
    def csv_list(self) -> Union[None, List[List[str]]]:
        if self.__csv_list is not None:
            return self.__csv_list

        csv_list = list()   # type: List[List[str]]
        csv_list.append(["Access Point Combination: " + self.id])
        csv_list.append(["Zones"] + [str(x) for x in self.__Vector_Dict.keys()])

        for measured_zone, vector in self.measured_zones_and_vectors:
            csv_list.append([str(measured_zone)] + [x for x in vector.values()])

        self.__csv_list = csv_list
        return csv_list
    # endregion

    # region Setters
    @id.setter
    def id(self, id: str) -> None:
        self.__id = id
    # endregion

    # region Getters
    def get_value(self, measured_zone: Zone, actual_zone: Zone) -> float:
        return self.__Vector_Dict[measured_zone][actual_zone]

    def get_vector(self, measured_zone: Zone) -> Dict[Zone, float]:
        return self.__Vector_Dict[measured_zone]

    def get_vector_sum(self, measured_zone: Zone) -> float:
        return sum(self.__Vector_Dict[measured_zone])
    # endregion

    # region Setters
    def set_value(self, measured_zone: Zone, actual_zone: Zone, value: float) -> None:
        self.__Vector_Dict[measured_zone][actual_zone] = value
    # endregion

    # region Incrementer
    def increment_value(self, measured_zone: Zone, actual_zone: Zone, increment_value: Union[int, float] = 1) -> None:
        self.__Vector_Dict[measured_zone][actual_zone] += increment_value
    # endregion

    # region Methods
    def scale_matrix(self, scalar: float) -> None:
        for vector in self.__Vector_Dict.values():
            for zone in vector.keys():
                vector[zone] *= scalar

    def print_matrix(self) -> None:
        print(self.id)
        for vector in self.__Vector_Dict.items():
            print(vector)

    def record_matrix(self, file_path: str) -> None:
        file_path += self.id + ".csv"

        with open(file_path, "w", newline='') as csvFile:
            writer = csv.writer(csvFile)

            for row in self.csv_list:
                writer.writerow(row)
    # endregion

    # region Static Methods
    @staticmethod
    def scalar_vector(vector: Dict[Zone, float], scalar: float) -> Dict[Zone, float]:
        return {zone: scalar * vector[zone] for zone in vector.keys()}
    # endregion
