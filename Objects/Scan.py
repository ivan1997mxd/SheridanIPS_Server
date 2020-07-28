from typing import Dict, List
from Objects.AccessPoint import AccessPoint


class Scan:

    def __init__(self, access_point_and_rssis: Dict[AccessPoint, int], zone: int, folder: str):
        self.__zone = zone
        self.__access_point_and_rssis = access_point_and_rssis
        self.__folder = folder

    @property
    def zone(self) -> int:
        return self.__zone

    @property
    def get_APs_and_RSSIs(self) -> Dict[AccessPoint, int]:
        return self.__access_point_and_rssis

    @property
    def rssis(self) -> List[int]:
        return self.__access_point_and_rssis.values()

    def get_rssi(self, ap: AccessPoint) -> int:
        return self.__access_point_and_rssis[ap]

    def __repr__(self):
        return "Zone: " + str(self.__zone) + " Scan Data : " + str(self.__access_point_and_rssis)
