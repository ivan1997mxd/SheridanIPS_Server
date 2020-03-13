from Resources.Objects.Matrices.NormalizedDistribution import build_normalized_distributions, sort_matrices
from Resources.Objects.Matrices.ProbabilityDistribution import *
from Resources.Objects.Matrices.CombinedDistribution import *
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint import GridPoint
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.TestData import create_test_data_list
from Resources.Objects.Zone import get_all_zones
from Algorithms.Combination.Combination import build_combined_distributions
from Algorithms.Combination.Combination import get_combination_function
from typing import Tuple, List, Callable
from time import time
import csv


def create_all_matrices_from_rssi_data(access_points: List[AccessPoint],
                                       centroids: List[Centroid],
                                       zones: List[Zone],
                                       training_data: List[Sample],
                                       testing_data: List[Sample],
                                       combination_method: Callable,
                                       num_combinations: int,
                                       skip_good_classifiers: bool = False) -> Tuple[List[ProbabilityMatrix],
                                                                                     List[NormalizedMatrix],
                                                                                     List[CombinedMatrix],
                                                                                     List[NormalizedMatrix],
                                                                                     Dict[NormalizedMatrix, TestResult]
]:
    start_time = time()

    # 4. Calculate distributions from RSSI scan data.
    probability_distributions = build_probability_distributions(access_points=access_points,
                                                                centroids=centroids,
                                                                zones=zones,
                                                                training_data=training_data)

    print("-- Completed Probability Distributions.")

    normalized_distributions = build_normalized_distributions(probability_distributions=probability_distributions)

    print("-- Completed Normalized Distributions.")

    object_creation_time = time()

    # 5. Start combination algorithm.

    combination_start_time = time()

    combined_distributions, normalized_combined_distributions = build_combined_distributions(
        normalized_distributions=normalized_distributions,
        training_data=training_data,
        skip_good_classifiers=skip_good_classifiers,
        centroids=centroids,
        zones=zones,
        num_combinations=num_combinations,
        combination_method=combination_method
        #       combination_method=get_combination_function("WGT")    #JC-01 use for debugging
    )

    print("-- Completed Combination Distributions.")

    combination_end_time = time()

    # 6. Run tests.
    test_results = test_combination_matrices(normalized_combined_distributions=normalized_combined_distributions,
                                             centroids=centroids,
                                             zones=zones,
                                             testing_data=testing_data,
                                             #                                            combination_method=get_combination_function("DBG"))    #JC-01 use the debugging method.
                                             combination_method=combination_method)

    print("-- Completed Testing Distributions.")

    test_end_time = time()

    # # 7. Record all Matrices separately.
    # for p in probability_distributions:
    #     p.record_matrix(probability_matrix_folder_path)
    #
    # for n in normalized_distributions:
    #     n.record_matrix(normalized_distribution_folder_path)
    #
    # for r in combined_distributions:
    #     r.record_matrix(combined_distribution_folder_path)
    #
    # for n in normalized_combined_distributions:
    #     n.record_matrix(normalized_combined_distribution_folder_path)
    #
    # # 8. Record Probability and Normalized Matrices together.
    # record_matrices(matrix_list=normalized_distributions,
    #                 file_path=probability_and_normalized_file_path)
    #
    # record_matrices(matrix_list=normalized_combined_distributions,
    #                 file_path=combined_distribution_and_normalized_file_path)
    #
    # print("Completed Recording all Matrices.")

    end_time = time()

    # print("{} Combinations, {:.2f} Split Run Time Stats:".format(num_combinations, split_value))
    # print("-- Object creation time: {}".format(object_creation_time - start_time))
    print("-- Combination run time: {}".format(combination_end_time - combination_start_time))
    # print("-- Matrix record time: {}".format(end_time - combination_end_time))
    print("-- Testing run time: {}".format(test_end_time - combination_end_time))
    print("-- Matrix Production run time: {}".format(end_time - start_time))
    print()

    # 9. Return Matrix objects to caller
    return probability_distributions, normalized_distributions, combined_distributions, normalized_combined_distributions, test_results


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
