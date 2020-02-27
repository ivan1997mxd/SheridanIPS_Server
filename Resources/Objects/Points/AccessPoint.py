from math import pow, log10
from Resources.Objects.Points.Point import Point
import csv


# region Class
class AccessPoint(Point):

    # region Constructor
    def __init__(self, rmac: str, num: int, x: float, y: float, z: float,
                 tx_pwr1: float, tx_pwr2: float, pl_ref: float,
                 gain_rx: int = 2, s_variance: int = 2):

        super(AccessPoint, self).__init__(id=rmac, num=num, x=x, y=y, z=z)

        self.__tx_pwr1 = tx_pwr1
        self.__tx_pwr2 = tx_pwr2
        self.__gain_rx = gain_rx
        self.__s_variance = s_variance
        self.__plref = pl_ref
    # endregion

    # region Methods
    def ple_distance(self, rssi: int, pleValue: float) -> float:
        numerator = self.__tx_pwr1 - rssi + self.__gain_rx - self.__plref + self.__gain_rx + self.__s_variance
        denominator = 10 * pleValue
        return pow(10, numerator / denominator)

    def calculate_ple(self, distance: float, rssi: int) -> float:
        numerator = self.__tx_pwr1 - rssi + self.__gain_rx - self.__plref + self.__gain_rx + self.__s_variance
        denominator = 10 * log10(distance)
        return numerator / denominator
    # endregion

    # region Method Overrides
    def _reset(self):
        # There is nothing to reset.
        pass

    @classmethod
    def create_point_list(cls, file_path: str, *args, **kwargs) -> list:
        assert len(args) == 0, "Access Points are unable to accept variable arguments."
        assert len(kwargs) == 0, "Access Points are unable to accept keyword arguments."
        points = list()

        with open(file_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")

            for row_num, line in enumerate(readCSV):
                if row_num == 0:
                    continue

                id = row_num
                rmac = str(line[0])
                x_val = float(line[1])
                y_val = float(line[2])
                z_val = float(line[3])
                txPwr1 = float(line[4])
                txPwr2 = float(line[5])
                plref = float(line[6])

                points.append(AccessPoint(rmac=rmac, num=id, x=x_val, y=y_val, z=z_val,
                                          tx_pwr1=txPwr1, tx_pwr2=txPwr2, pl_ref=plref))

        return points
    # endregion

# endregion
