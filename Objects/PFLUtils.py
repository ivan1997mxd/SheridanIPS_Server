import math
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint


# This is a utility class that is used for PFL-related calculations.
class PFLUtils:
    # The constructor sets the transmission and receive gains and the
    # s-variance value.
    # Refer to the PFL paper for more information.
    def __init__(self):
        self.GAIN_TX = 2
        self.GAIN_RX = 2
        self.S_VARIANCE = 2

    # This method will calculate the distance between a grid point and an AP.
    def getAccessPointDistance(self, gx, gy, ax, ay):
        valuesX = math.pow(gx - ax, 2)
        valuesY = math.pow(gy - ay, 2)
        distance = math.sqrt(valuesX + valuesY)
        return distance

    # This method will calculate the distance value based off a given PLe value.
    # This is used during the online stage.
    # Refer to the paper for more information.
    def calculateDistanceFromPle(self, avgRssi, txPower, plRef, ple):
        # print("TxPower: ", txPower)
        # print("plRef: ", plRef)
        # print("avgRssi: ", avgRssi)
        # print("PLE: ", ple)
        top = txPower - avgRssi + self.GAIN_TX - plRef + self.GAIN_RX + self.S_VARIANCE
        bottom = 10 * ple
        divide = top / bottom
        result = math.pow(10, divide)
        return result

    # This method will calculate the PLe value based on the real distance between
    # the AP and a given grid point.
    # This is used during the offline stage.
    # Refer to the paper for more information.
    def calculatePleFromDistance(self, distance, txPower, plRef, avgRssi):
        # print("TxPower: ", txPower)
        # print("plRef: ", plRef)
        # print("avgRssi: ", avgRssi)
        top = txPower - avgRssi + self.GAIN_TX - plRef + self.GAIN_RX + self.S_VARIANCE
        bottom = 10 * math.log10(distance)
        result = top / bottom
        return result

    # This is used to calculate the distance between a grid point and the central
    # point while calculating the distance threshold of the DFL algorithm.
    def calculateDistanceBetweenPFLs(self, cx, cy, centroidx, centroidy):
        x = math.pow(cx - centroidx, 2)
        y = math.pow(cy - centroidy, 2)
        result = math.sqrt(x + y)
        return result
