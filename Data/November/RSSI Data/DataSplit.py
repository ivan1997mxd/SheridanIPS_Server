from src.Resources.Scripts.FolderCheck import check_and_create_folder
from os.path import isfile, join
from math import ceil, floor
from os import listdir
import random
import time
import csv


online_pct = .7


# TODO: Write a better sorting algorithm.
def split_test_data(online_pct: float = 0.5) -> None:

    # raise Exception("This method should not be run unless there is new scan data. The existing data is already sorted.")

    dates = ["November 19", "November 20", "November 21", "November 23"]
    times = ["15_00", "17_00", "19_00"]
    point_types = ["Center"]

    for date in dates:

        for time in times:

            for point_type in point_types:

                raw_data_source_path = "./Kalman-Filtered/{}/{}/{}/".format(date, time, point_type)
                sorted_offline_path = "./Test Data/Offline/{}/{}/{}/".format(date, time, point_type)
                sorted_online_path = "./Test Data/Online/{}/{}/{}/".format(date, time, point_type)

                check_and_create_folder(sorted_offline_path)
                check_and_create_folder(sorted_online_path)

                data_files = [f for f in listdir(raw_data_source_path) if isfile(join(raw_data_source_path, f))]

                for file in data_files:
                    # Vars for file output:
                    file_name = file
                    online_datums = list()
                    offline_datums = list()

                    with open(raw_data_source_path + file) as csvFile:
                        readCSV = csv.reader(csvFile, delimiter=",")

                        for line, scan in enumerate(readCSV):
                            # Each line holds one Access Point's worth of a scan.

                            num_rssis = len(scan) - 2
                            num_online = floor(num_rssis * online_pct)
                            num_offline = num_rssis - num_online

                            assert(num_online > 0), "FILE: {} SCAN: NUM RSSIS: {}, NUMONLINE: {}".format(raw_data_source_path + file, scan, num_rssis, num_online)
                            #assert(num_rssis - num_online > 0), "FILE: {} SCAN: NUM RSSIS: {}, NUMONLINE: {}".format(raw_data_source_path + file, scan, num_rssis, num_online)

                            bssid = scan[0]
                            ssid = scan[1]
                            rssis = [int(x) for x in scan[2:]]

                            random.shuffle(rssis)
                            random.shuffle(rssis)

                            online_rssis = rssis[0:num_online - 1]
                            offline_rssis = rssis[num_online:]

                            online_datums.append((bssid, ssid, online_rssis))
                            offline_datums.append((bssid, ssid, offline_rssis))

                    # Create the output files:
                    with open(sorted_online_path + file_name, "w", newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        for values in online_datums:
                            row = [values[0], values[1]]    # bssid and ssid
                            row += [x for x in values[2]]   # rssis
                            writer.writerow(row)

                    with open(sorted_offline_path + file_name, "w", newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        for values in offline_datums:
                            row = [values[0], values[1]]    # bssid and ssid
                            row += [x for x in values[2]]   # rssis
                            writer.writerow(row)


start_time = time.time()
split_test_data(online_pct=online_pct)
end_time = time.time()

print("Split time: {}s".format(end_time - start_time))
