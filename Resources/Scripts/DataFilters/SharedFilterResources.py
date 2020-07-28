from typing import List


rmacs = ["10:bd:18:c6:b7:a0", "70:0f:6a:c8:29:40", "70:0f:6a:de:ba:60", "70:0f:6a:de:c1:40"]
dates = ["November 19", "November 20", "November 21", "November 23"]
times = ["15_00", "17_00", "19_00"]
point_types = ["Center of Zone of Zone of Zone of Zone of Zone", "Grid Point", "Off-Center of Zone of Zone of Zone of Zone of Zone"]


class __ScanValue:
    def __init__(self, bssid: str, ssid: str):
        self.__bssid = bssid
        self.__ssid = ssid
        self.__rssis = list()   # type: List[int]

    @property
    def bssid(self) -> str:
        return self.__bssid

    @property
    def ssid(self) -> str:
        return self.__ssid

    @property
    def rssis(self) -> List[int]:
        return self.__rssis

    @rssis.setter
    def rssis(self, value: int) -> None:
        self.__rssis.append(value)

