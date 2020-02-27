from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix, build_normalized_distributions
from Resources.Objects.Matrices.ProbabilityDistribution import build_probability_distributions, ProbabilityMatrix
from Resources.Objects.Matrices.CombinedDistribution import CombinedMatrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.TestData import Sample, TestResult
from Resources.Objects.Zone import Zone, get_zone
from Algorithms.NearestNeighbour.NNv4 import get_NNv4
from typing import List, Dict, Tuple, Union, Callable
from math import log


def alpha(error: float) -> float:
    try:
        return 0.5 * (log(1 - error) - log(error))
    except ValueError:
        try:
            return 0.5 * (log((1 - error) / error))
        except ValueError:
            print(error)
            print((1 - error) / error)
            exit(4)


class __BoostingMatrix:

    def __init__(self, samples: List[Sample],
                 normalizations: List[NormalizedMatrix],
                 weight: float,
                 zones: List[Zone],
                 centroids: List[Centroid]):

        self.__correct_matrix = {s: {n: False for n in normalizations} for s in samples}    # type: Dict[Sample, Dict[NormalizedMatrix, bool]]
        self.__normalizations = normalizations
        self.__correct_weight = weight
        self.__incorrect_weight = weight
        self.__centroids = centroids
        self.__zones = zones

        # Calculate all the pass/fails upon instantiation:
        self.__test()

    # region Properties
    @property
    def best_classifier(self) -> Tuple[NormalizedMatrix, float]:
        return self.get_next_best(ignore_bests=None)
        # # TODO: There is a cleaner way to do this using min(attr=X). Figure it out later.
        # # Best Classifier is defined by the matrix with the lowest cumulative error rate.
        # lowest_error_matrix = self.__normalizations[0]
        # lowest_error = self.get_matrix_error(lowest_error_matrix)
        # for matrix in self.__normalizations[1:]:
        #     matrix_error = self.get_matrix_error(matrix)
        #     if matrix_error < lowest_error:
        #         lowest_error_matrix = matrix
        #         lowest_error = matrix_error
        # return lowest_error_matrix, lowest_error

    @property
    def correct_weight(self) -> float:
        return self.__correct_weight

    @property
    def incorrect_weight(self) -> float:
        return self.__incorrect_weight

    @property
    def sample_size(self) -> int:
        return len(self.__correct_matrix)
    # endregion

    # region Getters
    def get_correct_count(self, normalization: NormalizedMatrix) -> int:
        return sum([correct_dict[normalization] for correct_dict in self.__correct_matrix.values()])

    def get_matrix_error(self, n: NormalizedMatrix) -> float:
        return self.incorrect_weight * self.get_incorrect_count(n)

    def get_incorrect_count(self, normalization: NormalizedMatrix) -> int:
        return self.sample_size - self.get_correct_count(normalization)

    def get_next_best(self, ignore_bests: Union[None, List[Tuple[NormalizedMatrix, float]]]):
        # TODO: There is a cleaner way to do this using min(attr=X). Figure it out later.
        if ignore_bests is None:
            matrices_to_ignore = []
        else:
            matrices_to_ignore = [x[0] for x in ignore_bests]

        # Best Classifier is defined by the matrix with the lowest cumulative error rate.
        matrices_to_check = [x for x in self.__normalizations if x not in matrices_to_ignore]
        lowest_error_matrix = matrices_to_check[0]
        lowest_error = self.get_matrix_error(lowest_error_matrix)
        for matrix in matrices_to_check[1:]:
            matrix_error = self.get_matrix_error(matrix)
            if matrix_error < lowest_error:
                lowest_error_matrix = matrix
                lowest_error = matrix_error
        return lowest_error_matrix, lowest_error

    def is_correct(self, sample: Sample, normalization: NormalizedMatrix) -> bool:
        return self.__correct_matrix[sample][normalization]
    # endregion

    # region Setters
    def set_weights(self, correct_weight: float, incorrect_weight: float) -> None:
        self.__correct_weight = correct_weight
        self.__incorrect_weight = incorrect_weight
    # endregion

    def __test(self) -> None:
        for sample in self.__correct_matrix.keys():
            for matrix in self.__correct_matrix[sample].keys():
                ap_rssi_dict = sample.get_ap_rssi_dict(*matrix.access_points)

                coord = get_NNv4(centroid_points=self.__centroids, rssis=ap_rssi_dict)
                zone = get_zone(zones=self.__zones, co_ordinate=coord)

                # True/False for Pass/Fail
                self.__correct_matrix[sample][matrix] = zone == sample.answer


def create_matrices(access_points: List[AccessPoint],
                    centroids: List[Centroid],
                    zones: List[Zone],
                    training_data: List[Sample],
                    testing_data: List[Sample],
                    combination_method: Callable,
                    num_combinations: int) -> Tuple[List[ProbabilityMatrix],
                                                    List[NormalizedMatrix],
                                                    List[CombinedMatrix],
                                                    List[NormalizedMatrix],
                                                    Dict[NormalizedMatrix, TestResult]]:

    # Step 1: Create probability distributions.
    probability_distributions = build_probability_distributions(access_points=access_points,
                                                                centroids=centroids,
                                                                zones=zones,
                                                                training_data=training_data)

    # Step 2: Normalize the distributions.
    normalized_distributions = build_normalized_distributions(probability_distributions=probability_distributions)

    # Step 3: Create a list of Distribution Trackers containing each Sample and each Normalization.
    boosting_tracker = __BoostingMatrix(samples=training_data,
                                        normalizations=normalized_distributions,
                                        weight=1/len(training_data),
                                        zones=zones,
                                        centroids=centroids)
    # Upon instantiation of the __BoostingMatrix, all pass/fails are automatically calculated.

    # Step 4: Retrieve the best (lowest error) Matrix.
    best_matrix, error = boosting_tracker.best_classifier
    best_alpha = alpha(error=error)

    # Step 5: Reset weights.
    correct_weight = 0.5 / boosting_tracker.get_correct_count(best_matrix)
    incorrect_weight = 0.5 / boosting_tracker.get_correct_count(best_matrix)
    boosting_tracker.set_weights(correct_weight=correct_weight, incorrect_weight=incorrect_weight)

    # Step 6: Get the num_combinations of other best matrices:
    best_found = [(best_matrix, best_alpha)]                  # type: List[Tuple[NormalizedMatrix, float]]
    for _ in range(2, num_combinations + 1):
        next_best_matrix, next_best_error = boosting_tracker.get_next_best(ignore_bests=best_found)

        # Reset weights.
        correct_weight = 0.5 / boosting_tracker.get_correct_count(next_best_matrix)
        incorrect_weight = 0.5 / boosting_tracker.get_correct_count(next_best_matrix)
        boosting_tracker.set_weights(correct_weight=correct_weight, incorrect_weight=incorrect_weight)

        # Store next best matrix:
        best_found.extend([(next_best_matrix, alpha(next_best_error))])

    # Step 7: Scale each matrix in best_found by their alpha values.
    for tup in best_found:
        tup[0].scale_matrix(tup[1])
    scaled_matrices = [x[0] for x in best_found]

    # Step 8: Combine the matrices.
    combined_matrix = CombinedMatrix(*scaled_matrices, size=scaled_matrices[0].size)

    # For each vector in the primary matrix:
    for measured_zone, vector in scaled_matrices[0].measured_zones_and_vectors:

        new_vector = {m: v for m, v in vector.items()}

        # For every over matrix:
        for matrix in scaled_matrices[1:]:

            # Get the matching vector:
            secondary_vector = matrix.get_vector(measured_zone)

            # Sum them:
            for actual_zone, value in secondary_vector.items():

                new_vector[actual_zone] += value

        # Update combined_matrix:
        for actual_zone, value in new_vector.items():
            combined_matrix.set_value(measured_zone=measured_zone, actual_zone=actual_zone, value=value)

    # Step 9: Normalize the new matrix.
    normalized_combination = NormalizedMatrix(matrix=combined_matrix, combine_ids=True)

    # Step 10: Test.
    test_results = test_combined_matrix(normalized_combination=normalized_combination,
                                        centroids=centroids,
                                        zones=zones,
                                        testing_data=testing_data,
                                        combination_method=combination_method)

    # 11. Return Matrix objects to caller
    return probability_distributions, normalized_distributions, [combined_matrix], [normalized_combination], {normalized_combination: test_results}


def test_combined_matrix(normalized_combination: NormalizedMatrix,
                         centroids: List[Centroid],
                         zones: List[Zone],
                         testing_data: List[Sample],
                         combination_method: Callable) -> TestResult:

    combine_vectors = combination_method

    test_result = TestResult()

    resultant = normalized_combination.parent_matrix

    for sample in testing_data:
        vectors = list()        # type: List[Dict[Zone, float]]

        for distribution in resultant.normalizations:

            ap_rssi_dict = sample.get_ap_rssi_dict(*distribution.access_points)

            coord = get_NNv4(centroid_points=centroids, rssis=ap_rssi_dict)
            zone = get_zone(zones=zones, co_ordinate=coord)

            vector = distribution.get_vector(zone)
            vectors.append(vector)

        resultant_vector = combine_vectors(sample.answer, *vectors)
        test_result.record(sample.answer, resultant_vector)

    return test_result
