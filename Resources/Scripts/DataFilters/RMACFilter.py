from .SharedFilterResources import __ScanValue
from typing import List, Dict
import csv


def rmac_filter_data(raw_data_file_path: str, rmacs: List[str]) -> Dict[str, __ScanValue]:

    # raise Exception("This method should not be run unless there is new scan data. The existing data is already sorted.")

    scan_dict = dict()  # type: Dict[str, __ScanValue]

    with open(raw_data_file_path) as csvFile:
        readCSV = csv.reader(csvFile, delimiter=",")

        for line, scan in enumerate(readCSV):

            if scan[0] not in rmacs:
                continue

            key = scan[0] + "-" + scan[1]

            if key in scan_dict.keys():
                for rssi in scan[2:]:
                    scan_dict[key].rssis = int(rssi)
            else:
                scanvalue = __ScanValue(scan[0], scan[1])
                for rssi in scan[2:]:
                    scanvalue.rssis = int(rssi)
                scan_dict[key] = scanvalue

    return scan_dict

    # with open(filtered_data_path + file, "w", newline='') as csvFile:
    #     writer = csv.writer(csvFile)
    #
    #     for scan in scan_dict.values():
    #         row = [scan.bssid, scan.ssid]
    #         row += [x for x in scan.rssis]
    #
    #         writer.writerow(row)

