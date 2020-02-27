from Resources.Objects.Matrices.Matrix import Matrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.Centroid import Centroid
from Algorithms.NearestNeighbour.NNv4 import get_NNv4
from Resources.Objects.Zone import Zone, get_zone
from Resources.Objects.TestData import Sample
from typing import List, Dict, Tuple
import itertools
import csv


class ProbabilityMatrix(Matrix):
    # TODO: This object no longer has any real functionality. Should be removed in the future.

    def __init__(self, access_points: List[AccessPoint], zones: List[Zone], size):
        super(ProbabilityMatrix, self).__init__(access_points=access_points, zones=zones, size=size)


# region External Constructor
def build_probability_distributions(access_points: List[AccessPoint],
                                    centroids: List[Centroid],
                                    zones: List[Zone],
                                    training_data: List[Sample]) -> List[ProbabilityMatrix]:

    probability_matrices = list()   # type: List[ProbabilityMatrix]

    # For every combination of Access Points, create one Probability Matrix.
    for access_point_tuple in __get_access_point_combinations(access_points):

        probability_matrix = ProbabilityMatrix(access_points=[*access_point_tuple], zones=zones, size=len(zones))

        # For every zone file:
        for zone in zones:

            zone_appearance_tracker = dict()  # type: Dict[Zone, int]

            samples_taken = 0  # type: int

            for sample in training_data:

                if sample.answer is not zone:
                    continue

                # Get a dictionary of the AP: RSSI we want.
                ap_rssi_dict = sample.get_ap_rssi_dict(*access_point_tuple)

                # Calculate the zone, using the dictionary just created.
                calculated_co_ordinate = get_NNv4(centroid_points=centroids, rssis=ap_rssi_dict)
                calculated_zone = get_zone(zones, calculated_co_ordinate)

                # Update the appearance tracker.
                if calculated_zone in zone_appearance_tracker:
                    zone_appearance_tracker[calculated_zone] += 1
                else:
                    zone_appearance_tracker[calculated_zone] = 1

                # Update the samples tracker
                samples_taken += 1

            # Calculate the column of the Probability Matrix:
            for measured_zone, count in zone_appearance_tracker.items():
                # P(E | H) = (Number of times appeared + (1 / sample size)) / (sample size + 1)
                probability = (count * samples_taken + 1) / (samples_taken * (samples_taken + 1))
                probability_matrix.set_value(
                    measured_zone=measured_zone,
                    actual_zone=zone,
                    value=probability)

        probability_matrices.append(probability_matrix)

    return probability_matrices
# endregion


# region File Specific Private Methods
def __get_rssis_from_offline_zone_files(zone: Zone, rssi_folder_path: str) -> List[int]:
    rssi_folder_path += "Zone " + str(zone.num) + ".csv"

    with open(rssi_folder_path, "r", newline='') as csvFile:
        reader = csv.reader(csvFile)
        for values in reader:
            yield [int(x) for x in values]


def __get_access_point_combinations(access_points: List[AccessPoint]) -> Tuple[AccessPoint, ...]:
    for i in range(0, len(access_points) + 1):
        for subset in itertools.combinations(access_points, i):
            if len(subset) < 2:
                continue
            yield subset
# endregion

