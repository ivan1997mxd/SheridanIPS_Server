import math
from statistics import mean

from kombu.utils import nested
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint, find_cloest_p
from Resources.Objects.Zone import Zone
from os.path import join, isfile
from os import listdir
from math import floor
from typing import List, Dict
import pymongo
import csv
from flask_pymongo import PyMongo

client = pymongo.MongoClient("localhost", 27017)
db = client.SheridanIPS


class Sample:
    """Each Sample consists of a Scan Data Dictionary, and the Zone Answer.

    The Scan Data Dictionary holds 1 set of Scan values - Each Access Point seen, and it's respective RSSI value.
    The Zone Answer is the Zone that the actual Scan was taken from.
    """

    def __init__(self, actual_zone: Zone, scan_data: Dict[AccessPoint, int], random_coord: list = None):
        self.__actual_zone = actual_zone  # type: Zone
        self.__scan_data = scan_data  # type: Dict[AccessPoint, int]

        # Used in Adaptive Boosting methods:
        self.__weight = float()  # type: float
        self.__correct = bool()  # type: bool
        self.random_coord = random_coord

    # region Properties
    @property
    def answer(self) -> Zone:
        return self.__actual_zone

    @property
    def scan(self) -> Dict[AccessPoint, int]:
        return self.__scan_data

    @property
    def weight(self) -> float:
        return self.__weight

    @property
    def correct(self) -> bool:
        return self.__correct

    @property
    def coord(self) -> list:
        return self.random_coord

    # endregion

    # region Setters
    @weight.setter
    def weight(self, value: float) -> None:
        self.__weight = value

    @correct.setter
    def correct(self, value: bool) -> None:
        self.__correct = value

    # endregion

    def rssi(self, access_point: AccessPoint) -> int:
        return self.__scan_data[access_point]

    def get_ap_rssi_dict(self, *access_points: AccessPoint) -> Dict[AccessPoint, int]:
        return {k: v for k, v in self.__scan_data.items() if k in access_points}

    def get_rssis(self, *access_points: List[AccessPoint]) -> List[int]:
        return [rssi for ap, rssi in self.__scan_data.items() if ap in access_points]

    def __repr__(self) -> str:
        Str = "Zone: {} - ".format(self.__actual_zone.num)
        for k, v in self.__scan_data.items():
            Str += " {AP: " + str(k) + ": RSSI: " + str(v) + "}"
        return Str


class NewSample:
    """Each Sample consists of a Scan Data Dictionary, and the Zone Answer.

    The Scan Data Dictionary holds 1 set of Scan values - Each Access Point seen, and it's respective RSSI value.
    The Zone Answer is the Zone that the actual Scan was taken from.
    """

    def __init__(self, scan_data: Dict[AccessPoint, int], random_coord: list = None, actual_gp: GridPoint = None):
        self.__actual_gp = actual_gp  # type: GridPoint
        self.__scan_data = scan_data  # type: Dict[AccessPoint, int]
        self.__random_coord = random_coord

        # Used in Adaptive Boosting methods:
        self.__weight = float()  # type: float
        self.__correct = bool()  # type: bool

    # region Properties
    @property
    def answer(self) -> GridPoint:
        return self.__actual_gp

    @property
    def scan(self) -> Dict[AccessPoint, int]:
        return self.__scan_data

    @property
    def weight(self) -> float:
        return self.__weight

    @property
    def correct(self) -> bool:
        return self.__correct

    @property
    def coord(self) -> list:
        return self.__random_coord

    # endregion

    # region Setters
    @weight.setter
    def weight(self, value: float) -> None:
        self.__weight = value

    @correct.setter
    def correct(self, value: bool) -> None:
        self.__correct = value

    # endregion

    def rssi(self, access_point: AccessPoint) -> int:
        return self.__scan_data[access_point]

    def get_ap_rssi_dict(self, *access_points: AccessPoint) -> Dict[AccessPoint, int]:
        return {k: v for k, v in self.__scan_data.items() if k in access_points}

    def get_rssis(self, *access_points: List[AccessPoint]) -> List[int]:
        return [rssi for ap, rssi in self.__scan_data.items() if ap in access_points]

    def __repr__(self) -> str:
        Str = "GP: {} - ".format(self.__actual_gp.num)
        for k, v in self.__scan_data.items():
            Str += " {AP: " + str(k) + ": RSSI: " + str(v) + "}"
        return Str


class TestResult:

    def __init__(self):
        self.__correct = 0  # type: int
        self.__total_tests = 0  # type: int
        self.__sec_correct = 0  # type: int
        self.__answer_details = dict()  # type: Dict[Zone, Dict[str, int]]

    @property
    def answer_details(self):
        return self.__answer_details

    @property
    def accuracy(self) -> float:
        if self.__total_tests == 0:
            return 0
        return self.__correct / self.__total_tests

    @property
    def num_correct(self) -> int:
        return self.__correct

    @property
    def tests_ran(self) -> int:
        return self.__total_tests

    # region Comparison Operators
    def __lt__(self, other):
        return True if self.accuracy < other.accuracy else False

    def __le__(self, other):
        return True if self.accuracy <= other.accuracy else False

    def __gt__(self, other):
        return True if self.accuracy > other.accuracy else False

    def __ge__(self, other):
        return True if self.accuracy >= other.accuracy else False

    # endregion

    # JC-01 - add code to support 2nd guess information
    def record(self, zone: Zone, vector: Dict[Zone, float]) -> None:
        self.__total_tests += 1

        # Get max probability in the vector
        most_likely_zone = max(vector, key=vector.get)
        base_zone_error_1 = abs(most_likely_zone.num - zone.num)
        if most_likely_zone == zone:
            self.__correct += 1

            if zone not in self.__answer_details:
                self.__answer_details[zone] = {"times_tested": 1, "times_correct": 1, "times_2nd_correct": 0,
                                               "base_zone_error_1": 0.0, "base_zone_error_2": 0.0}

            else:
                self.__answer_details[zone]["times_tested"] += 1
                self.__answer_details[zone]["times_correct"] += 1
            return

        vector[most_likely_zone] = 0.0
        most_likely_zone = max(vector, key=vector.get)
        base_zone_error_2 = abs(most_likely_zone.num - zone.num)
        if most_likely_zone == zone:
            self.__sec_correct += 1

            if zone not in self.__answer_details:
                self.__answer_details[zone] = {"times_tested": 1, "times_correct": 0, "times_2nd_correct": 1,
                                               "base_zone_error_1": base_zone_error_1, "base_zone_error_2": 0.0}
            else:
                self.__answer_details[zone]["base_zone_error_1"] += base_zone_error_1
                self.__answer_details[zone]["times_tested"] += 1
                self.__answer_details[zone]["times_2nd_correct"] += 1
            return

        if zone not in self.__answer_details:
            self.__answer_details[zone] = {"times_tested": 1, "times_correct": 0, "times_2nd_correct": 0,
                                           "base_zone_error_1": base_zone_error_1,
                                           "base_zone_error_2": base_zone_error_2}
        else:
            self.__answer_details[zone]["base_zone_error_1"] += base_zone_error_1
            self.__answer_details[zone]["base_zone_error_2"] += base_zone_error_2
            self.__answer_details[zone]["times_tested"] += 1


def create_test_data_list(access_points: List[AccessPoint],
                          zones: List[Zone],
                          folder_path: str,
                          dates: List[str],
                          times: List[str]) -> List[Sample]:
    """This method returns a list of Samples.

        If one of the Access Points has returned fewer scans than the others, the average
        will be used to fill in the missing values.
    """

    samples = list()  # type: List[Sample]

    for date in dates:
        for time in times:

            sub_folder = "{}/{}/{}/Center of Zone/".format(folder_path, date, time)

            data_files = [f for f in listdir(sub_folder) if isfile(join(sub_folder, f))]

            for file in data_files:

                # Read the entire file, and load them into the dict.
                bssid_rssi_dict = dict()  # type: Dict[str, List[int]]
                answer = zones[int(file[-5]) - 1]
                longest_rssis = 0

                with open(sub_folder + file, "r", newline='') as csvFile:
                    reader = csv.reader(csvFile)

                    for scan in reader:

                        bssid = scan[0]
                        rssis = [int(x) for x in scan[1:]]

                        # This is no longer necessary since the scans were RMAC-filtered previously.
                        if bssid in bssid_rssi_dict.keys():
                            bssid_rssi_dict[bssid] += rssis
                        else:
                            bssid_rssi_dict[bssid] = rssis

                        if len(rssis) > longest_rssis:
                            longest_rssis = len(rssis)
                for index in range(longest_rssis):

                    ap_rssi_dict = dict()  # type: Dict[AccessPoint, int]
                    ap_rssi_dict_str = dict()  # type: Dict[str, int]

                    for key, rssis in bssid_rssi_dict.items():

                        if len(rssis) == 0:
                            continue

                        found = False

                        for access_point in access_points:

                            if key == access_point.id:
                                try:
                                    ap_rssi_dict[access_point] = rssis[index]
                                    ap_rssi_dict_str[access_point.id] = rssis[index]
                                except IndexError:
                                    # Hit because this AP may not have enough RSSI values. Append the average.
                                    ap_rssi_dict[access_point] = floor(sum(rssis) / len(rssis))
                                    ap_rssi_dict_str[access_point.id] = floor(sum(rssis) / len(rssis))
                                found = True
                                break
                            else:
                                pass

                    samples.append(Sample(answer, ap_rssi_dict))

    return samples


def calculate(inputValues: list, initialVariance: float, noise: float):
    kalmanGain = 0.0
    processNoise = noise
    variance_value = initialVariance
    measurementNoise = variance(inputValues)
    mean = inputValues[0]
    filtered_list = []

    for value in inputValues:
        variance_value = variance_value + processNoise
        kalmanGain = variance_value / (variance_value + measurementNoise)
        mean = mean + kalmanGain * (value - mean)
        variance_value = variance_value - (kalmanGain * variance_value)
        filtered_list.append(round(mean))
    return filtered_list


def variance(values: list) -> float:
    sum = 0.0
    average = mean(values)
    for v in values:
        sum += math.pow(v - average, 2)
    return sum / (len(values) - 1)


def create_from_db(access_points: List[AccessPoint],
                   zones: List[Zone], offline_data: list, random_coords: list = None):
    samples = list()  # type: List[Sample]
    for i, z in enumerate(offline_data):
        if random_coords is not None:
            zone_coords = random_coords[i]
        zone_data = z['zone_data']
        zone_num = z['zone_num']
        num = int(zone_num[6:])
        # if num > 6:
        #     num -= 6
        answer = zones[num - 1]
        for index, zd in enumerate(zone_data):
            if random_coords is not None:
                random_coord = zone_coords[index]
            bssid_rssi_dict = dict()  # type: Dict[str, List[int]]
            longest_rssis = 0
            for bssid, rssis in zd.items():
                # rssis = calculate(rssis, 50.0, 0.008)
                bssid_rssi_dict[bssid] = rssis
                if len(rssis) > longest_rssis:
                    longest_rssis = len(rssis)
            for index in range(longest_rssis):

                ap_rssi_dict = dict()  # type: Dict[AccessPoint, int]
                for access_point in access_points:
                    rssis = bssid_rssi_dict[access_point.id]
                    try:
                        rssi = rssis[index]
                    except IndexError:
                        rssi = floor(sum(rssis) / len(rssis))
                    ap_rssi_dict[access_point] = rssi

                if random_coords is not None:
                    samples.append(Sample(answer, ap_rssi_dict, random_coord))
                else:
                    samples.append(Sample(answer, ap_rssi_dict))
    return samples


def create_from_db_new(access_points: List[AccessPoint], grid_points: List[GridPoint], offline_data: list,
                       random_coords: list = None):
    samples = list()  # type: List[NewSample]
    for i, z in enumerate(offline_data):
        zone_data = z['zone_data']
        zone_num = int(z['zone_num'])
        if random_coords is not None:
            coord = random_coords[zone_num - 1]
            answer = find_cloest_p(grid_points, coord)
        else:
            answer = grid_points[zone_num - 1]
        bssid_rssi_dict = dict()  # type: Dict[str, List[int]]
        longest_rssis = 0
        for bssid, rssis in zone_data.items():
            # rssis = calculate(rssis, 50.0, 0.008)
            bssid_rssi_dict[bssid] = rssis
            if len(rssis) > longest_rssis:
                longest_rssis = len(rssis)
        for index in range(longest_rssis):

            ap_rssi_dict = dict()  # type: Dict[AccessPoint, int]

            for key, rssis in bssid_rssi_dict.items():
                if len(rssis) == 0:
                    continue

                for access_point in access_points:

                    if key == access_point.id:
                        try:
                            ap_rssi_dict[access_point] = rssis[index]
                        except IndexError:
                            # Hit because this AP may not have enough RSSI values. Append the average.
                            ap_rssi_dict[access_point] = floor(sum(rssis) / len(rssis))
                        break
                    else:
                        pass
            if random_coords is not None:
                samples.append(NewSample(scan_data=ap_rssi_dict, actual_gp=answer, random_coord=coord))
            else:
                samples.append(NewSample(scan_data=ap_rssi_dict, actual_gp=answer))

    return samples
