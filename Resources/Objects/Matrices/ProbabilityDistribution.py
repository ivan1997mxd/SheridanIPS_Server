from Algorithms.NearestNeighbour.Calculation import get_calculation_function
from Objects.Collector import IndividualModel
from Resources.Objects.Matrices.Matrix import Matrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Algorithms.NearestNeighbour.NNv4 import get_NNv4_RSSI, get_NNv4
from Resources.Objects.Zone import Zone, get_zone
from Resources.Objects.TestData import Sample
from typing import List, Dict, Tuple, Callable
import itertools
import csv


class ProbabilityMatrix(Matrix):
    # TODO: This object no longer has any real functionality. Should be removed in the future.

    def __init__(self, access_points: List[AccessPoint], zones: List[Zone], size):
        super(ProbabilityMatrix, self).__init__(access_points=access_points, zones=zones, size=size)


def build_svm_probability_distribution(trained_model: IndividualModel) -> ProbabilityMatrix:
    return build_svm_probability_distributions([trained_model])[0]


def build_svm_probability_distributions(trained_models: List[IndividualModel]) -> List[ProbabilityMatrix]:
    probability_matrices = list()  # type: List[ProbabilityMatrix]

    for model in trained_models:

        zones = model.zones
        p_dist = ProbabilityMatrix(model.access_points, zones, len(zones))

        for zone in zones:

            zone_appearance_tracker = dict()  # type: Dict[Zone, int]
            # predicted_versus_actual = list()  # type: List[Tuple[int, int]]
            samples_taken = 0
            for index, class_num in enumerate(model.test_classes):
                test_class = zones[class_num - 1]
                if test_class == zone:
                    prediction = int(model.predictions[index])
                    pred_zone = zones[prediction - 1]
                    if pred_zone in zone_appearance_tracker:
                        zone_appearance_tracker[pred_zone] += 1
                    else:
                        zone_appearance_tracker[pred_zone] = 1

                    # Update the samples tracker
                    samples_taken += 1

            # Calculate the column of the Probability Matrix:
            for measured_zone, count in zone_appearance_tracker.items():
                # P(E | H) = (Number of times appeared + (1 / sample size)) / (sample size + 1)
                probability = (count * samples_taken + 1) / (samples_taken * (samples_taken + 1))
                p_dist.set_value(
                    measured_zone=measured_zone,
                    actual_zone=zone,
                    value=probability)

        probability_matrices.append(p_dist)

    return probability_matrices


# region External Constructor
def build_probability_distributions(access_points: List[AccessPoint],
                                    access_point_combinations: List[Tuple[AccessPoint, ...]],
                                    centroids: List[Centroid],
                                    grid_points: List[GridPoint],
                                    zones: List[Zone],
                                    training_data: List[Sample],
                                    location_mode: str) -> List[ProbabilityMatrix]:
    probability_matrices = list()  # type: List[ProbabilityMatrix]
    calculated_co_ordinate = Tuple[float, float]
    # For every combination of Access Points, create one Probability Matrix.
    for access_point_tuple in access_point_combinations:

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
                location_method = get_calculation_function(location_mode)
                if location_mode == "NNv4" or location_mode == "kNNv2" or location_mode == "kNNv1":
                    calculated_co_ordinate = location_method(centroid_points=centroids, rssis=ap_rssi_dict)
                if location_mode == "kNNv3":
                    calculated_co_ordinate = location_method(grid_points=grid_points, rssis=ap_rssi_dict)

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
