from Objects.Collector import IndividualModel
from Resources.Objects.Matrices.NormalizedDistribution import build_normalized_distributions, sort_matrices
from Resources.Objects.Matrices.ProbabilityDistribution import *
from Resources.Objects.Matrices.CombinedDistribution import *
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.TestData import create_test_data_list
from Resources.Objects.Zone import Zone
from Algorithms.Combination.Combination import build_combined_distributions
from Algorithms.Combination.Combination import get_combination_function
from typing import Tuple, List, Callable
from time import time
import csv


def create_all_svm_matrices(trained_models: List[IndividualModel]):
    start_time = time()
    probability_distributions = build_svm_probability_distributions(trained_models=trained_models)
    normalized_distributions = build_normalized_distributions(probability_distributions=probability_distributions)
    sort_matrices(normalized_distributions)
    end_time = time()
    print("-- SVM Matrix Production run time: {}".format(end_time - start_time))
    return probability_distributions, normalized_distributions


def create_all_matrices_from_rssi_data(access_points: List[AccessPoint],
                                       access_point_combinations: List[Tuple[AccessPoint, ...]],
                                       zones: List[Zone],
                                       training_data: List[Sample],
                                       combination_mode: str,
                                       location_mode: str,
                                       num_combinations: int,
                                       centroids: List[Centroid],
                                       grid_points: List[GridPoint],
                                       testing_data: List[Sample] = None,
                                       do_combination: bool = True):
    start_time = time()

    # 4. Calculate distributions from RSSI scan data.
    probability_distributions = build_probability_distributions(access_points=access_points,
                                                                access_point_combinations=access_point_combinations,
                                                                centroids=centroids,
                                                                grid_points=grid_points,
                                                                zones=zones,
                                                                training_data=training_data,
                                                                location_mode=location_mode)

    # print("-- Completed Probability Distributions.")

    normalized_distributions = build_normalized_distributions(probability_distributions=probability_distributions)

    # Sort the matrices:
    # sort_matrices(normalized_distributions)

    # print("-- Completed Normalized Distributions.")

    # 5. Start combination algorithm.
    if do_combination:

        combination_start_time = time()

        combined_distributions, normalized_combined_distributions = build_combined_distributions(
            normalized_distributions=normalized_distributions,
            training_data=training_data,
            centroids=centroids,
            grid_points=grid_points,
            zones=zones,
            num_combination=num_combinations,
            combination_mode=combination_mode,
            location_mode=location_mode
        )

        print("-- Completed Combination Distributions.")

        combination_end_time = time()

        # 6. Run tests.
        combination_method = get_combination_function(combination_mode)
        test_results = test_combination_matrices(normalized_combined_distributions=normalized_combined_distributions,
                                                 centroids=centroids,
                                                 zones=zones,
                                                 testing_data=testing_data,
                                                 #                                            combination_method=get_combination_function("DBG"))    #JC-01 use the debugging method.
                                                 combination_method=combination_method)



        print("-- Completed Testing Distributions.")
        test_end_time = time()
        end_time = time()

        # print("{} Combinations, {:.2f} Split Run Time Stats:".format(num_combinations, split_value))
        # print("-- Object creation time: {}".format(object_creation_time - start_time))
        print("-- Combination run time: {}".format(combination_end_time - combination_start_time))
        # print("-- Matrix record time: {}".format(end_time - combination_end_time))
        print("-- Testing run time: {}".format(test_end_time - combination_end_time))
        print("-- Matrix Production run time: {}".format(end_time - start_time))
        print()
        return probability_distributions, normalized_distributions, combined_distributions, normalized_combined_distributions, test_results
    else:
        jc_test_results = test_normalized_list(normalized_list=normalized_distributions,
                                               centroids=centroids,
                                               zones=zones,
                                               testing_data=testing_data)
        return probability_distributions, normalized_distributions, jc_test_results


def record_matrices(matrix_list: List[NormalizedMatrix], file_path: str):
    # Sort in ascending order of normalized matrix error:
    sort_matrices(matrix_list=matrix_list)

    with open(file_path, "w", newline='') as csvFile:
        writer = csv.writer(csvFile)

        for matrix in matrix_list:

            parent_matrix = matrix.parent_matrix

            max_index = 0
            for index, value in enumerate(parent_matrix.csv_list):
                if index == 0:
                    writer.writerow(value + ["" for _ in range(matrix.size + 2)] + matrix.csv_list[index])
                else:
                    writer.writerow(value + ["", ""] + matrix.csv_list[index])
                max_index = index

            writer.writerow(["" for _ in range(matrix.size + 3)] + matrix.csv_list[max_index + 1])

            writer.writerow([])
            writer.writerow([])
