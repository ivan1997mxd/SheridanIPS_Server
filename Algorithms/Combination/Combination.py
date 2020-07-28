import numpy as np

from Algorithms.NearestNeighbour.Calculation import get_calculation_function
from Algorithms.svm.svm import svm_model
from Algorithms.svm.svmutil import svm_predict, svm_train
from Resources.Objects.Points.AccessPoint import AccessPoint
from Objects.FinalCombinationContainer import FinalCombinationContainer
from Resources.Objects.Matrices.NormalizedDistribution import sort_matrices, NormalizedMatrix
from Resources.Objects.Matrices.CombinedDistribution import CombinedMatrix
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Algorithms.NearestNeighbour.NNv4 import get_NNv4, get_NNv4_RSSI
from Resources.Objects.Zone import Zone, get_zone
from Resources.Objects.TestData import Sample
from typing import List, Dict, Tuple, Callable, Iterator
from itertools import combinations
import math
import pymongo

theAnswer = 1


def __nCr(n: int, r: int):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def get_combination_function(combination_mode: str) -> Callable:
    if combination_mode == "AVG":
        return __sum_combine_vectors
    elif combination_mode == "WGT":
        return __weighted_combine_vectors
    elif combination_mode == "DBG":
        return __dbg_combine_vectors
    else:
        raise Exception("The Combination Mode: {} is invalid.".format(combination_mode))


def __ap_combinations(averaged_normalized_list: List[Tuple[NormalizedMatrix, List[svm_model]]],
                      num_combinations: int) -> Iterator[Tuple[AccessPoint, ...]]:
    normalized_distributions = [average_tuple[0] for average_tuple in averaged_normalized_list]
    access_points = [normalized_distribution.access_points_tuple for normalized_distribution in
                     normalized_distributions]
    primary_ap = tuple(access_points[0])
    combination = combinations(access_points[1:], num_combinations - 1)
    num_combinations = __nCr(len(normalized_distributions) - 1, num_combinations - 1)
    print("-- There will be {} matrices produced.".format(num_combinations))
    for combo in combination:
        yield [primary_ap] + [*combo]


def __combinations(normalized_distributions: List[NormalizedMatrix],
                   num_combinations: int) -> Iterator[List[NormalizedMatrix]]:
    # Get combinations of matrices:
    primary_distribution = normalized_distributions[0]
    combination = combinations(normalized_distributions[1:], num_combinations - 1)

    num_combinations = __nCr(len(normalized_distributions) - 1, num_combinations - 1)

    print("-- There will be {} matrices produced.".format(num_combinations))

    for combo in combination:
        yield [primary_distribution] + [*combo]


# JC Used for training phase
def build_combined_distributions(centroids: List[Centroid],
                                 grid_points: List[GridPoint],
                                 zones: List[Zone],
                                 normalized_distributions: List[NormalizedMatrix],
                                 training_data: List[Sample],
                                 combination_mode: str,
                                 location_mode: str,
                                 num_combination: int) -> Tuple[List[CombinedMatrix], List[NormalizedMatrix]]:
    calculated_co_ordinate = Tuple[float, float]
    combined_vector = Dict[Zone, float]

    # sort_matrices(normalized_distributions)

    # Set combination method:
    combine_vectors = get_combination_function(combination_mode)

    # Containers to hold results.
    combined_distributions = list()  # type: List[CombinedMatrix]
    normalized_combinations = list()  # type: List[NormalizedMatrix]

    # For every combination:
    for matrix_list in __combinations(normalized_distributions, num_combination):

        # Empty matrix to hold the results of all tests.
        combined_matrix = CombinedMatrix(*matrix_list, size=matrix_list[0].size)

        for sample in training_data:
            vectors = list()  # type: List[Dict[Zone, float]]
            answers = list()
            for matrix in matrix_list:
                ap_rssi_dict = sample.get_ap_rssi_dict(*matrix.access_points)
                location_method = get_calculation_function(location_mode)
                if location_mode == "NNv4" or location_mode == "kNNv2" or location_mode == "kNNv1":
                    calculated_co_ordinate = location_method(centroid_points=centroids, rssis=ap_rssi_dict)
                if location_mode == "kNNv3":
                    calculated_co_ordinate = location_method(grid_points=grid_points, rssis=ap_rssi_dict)
                # coord = get_NNv4(centroid_points=centroids,
                #                       rssis=sample.get_ap_rssi_dict(*matrix.access_points))
                zone = get_zone(zones=zones, co_ordinate=calculated_co_ordinate)
                # print("find at " + str(zone) + " in matrix " + str(matrix.id))
                vector = matrix.get_vector(zone)
                vectors.append(vector)
                answers.append(zone)

            if combination_mode == "WGT":
                # Get combined vector from combination of above vectors.
                combined_vector = combine_vectors(answers, *vectors)

            # Add resultant vector the the ResultantMatrix object.
            combined_matrix.increment_cell(sample.answer, combined_vector)

        # Normalize the resultant ResultantMatrix object:
        normalized_combination = NormalizedMatrix(combined_matrix, combine_ids=True)

        # Append to both container lists:
        combined_distributions.append(combined_matrix)
        normalized_combinations.append(normalized_combination)
        # OfflineData.insert(normalized_combination)

    return combined_distributions, normalized_combinations


def build_jc_combined_distributions(averaged_distributions: Dict[Tuple[AccessPoint, ...],
                                                                 Tuple[NormalizedMatrix, List[svm_model]]],
                                    num_combinations: int,
                                    training_data: List[Sample]
                                    ) -> List[FinalCombinationContainer]:
    # Set combination method:
    combine_vectors = __weighted_combine_vectors

    averaged_normalized_list = sorted(averaged_distributions.values(),
                                      key=lambda x: x[0].average_matrix_success, reverse=True)

    primary_matrix = averaged_normalized_list[0][0]
    primary_svm_list = averaged_normalized_list[0][1]
    access_point_tuple = primary_matrix.access_points_tuple
    zones = primary_matrix.zones

    # Containers to hold matrices being returned.
    final_combination_container = list()  # type: List[FinalCombinationContainer]
    combined_distributions = list()  # type: List[CombinedMatrix]
    normalized_combined_distributions = list()  # type: List[NormalizedMatrix]

    # Containers to hold data from the passed param.
    # access_point_combinations = list(averaged_distributions.keys())

    # amount = __nCr(len(averaged_distributions), num_combinations)
    # print("------ There will be {} combinations.".format(int(amount)))
    tracker = 1

    primary_predict_zone = list()
    primary_predict = list()
    ap_test_features = list()
    test_features = [data.scan for data in training_data]
    test_class = [data.answer.num for data in training_data]
    aps_being_used = [x for x in access_point_tuple]

    for feature_set in test_features:
        ap_test_features.append(
            [value for key, value in feature_set.items() if key in aps_being_used])
    for svm in primary_svm_list:
        # 通过找出最好的svm_model来进行online stage
        p_labs, p_acc, p_vals = svm_predict(y=test_class, x=ap_test_features, m=svm, options="-q")
        # print("the accuracy for primary is {}".format(p_acc[0]))
        primary_predict.append(p_labs)

    # my_combo = m.get_combination()
    # final_count = list()
    # for p in p_vals:
    #     count_dict = dict()
    #     for index in range(len(p)):
    #         big_one = classfications[index][0]
    #         small_one = classfications[index][1]
    #         if p[index] > 0:
    #             if big_one in count_dict.keys():
    #                 count_dict[big_one] += 1
    #             else:
    #                 count_dict[big_one] = 1
    #         else:
    #             if small_one in count_dict.keys():
    #                 count_dict[small_one] += 1
    #             else:
    #                 count_dict[small_one] = 1
    #     final_count.append(max(count_dict, key=count_dict.get))

    for d in range(len(primary_predict[0])):
        zone_predictions = list()
        for p in primary_predict:
            zone_predictions.append(p[d])
        best_predict_zone = max(zone_predictions, key=zone_predictions.count)
        predicted_zone = zones[int(best_predict_zone) - 1]
        primary_predict_zone.append(predicted_zone)
    # For every possible ap combination: # TODO: This should pick the best normalized dist, instead of combining all.
    for ap_combination in __ap_combinations(averaged_normalized_list, num_combinations):
        print("------ Combination number {}:".format(tracker))
        print("------ Combining {}.".format(ap_combination))

        features = list()
        secondary_predict_zone = list()

        # Get the normalizations being combined:
        normalizations = [norm[0] for norm in [averaged_distributions[ap] for ap in ap_combination]]

        # Create an empty combined distribution:
        combined_distribution = CombinedMatrix(*normalizations, size=normalizations[0].size)

        aps_being_used = normalizations[1].access_points_tuple
        svms = averaged_distributions[aps_being_used][1]
        matrix = averaged_distributions[aps_being_used][0]

        for feature_set in test_features:
            features.append(
                [value for key, value in feature_set.items() if key in ap_combination[1]])

        secondary_predict = list()
        for svm in svms:
            p_labs, p_acc, p_vals = svm_predict(y=test_class, x=features, m=svm, options="-q")
            # print("the accuracy for secondary is {}".format(p_acc[0]))
            secondary_predict.append(p_labs)

        for d in range(len(secondary_predict[0])):
            zone_predictions = list()
            for p in secondary_predict:
                zone_predictions.append(p[d])
            best_predict_zone = max(zone_predictions, key=zone_predictions.count)
            predicted_zone = zones[int(best_predict_zone) - 1]
            secondary_predict_zone.append(predicted_zone)

        for z in range(len(secondary_predict_zone)):
            secondary_vector = matrix.get_vector(secondary_predict_zone[z])
            primary_vector = primary_matrix.get_vector(primary_predict_zone[z])
            vectors = [primary_vector, secondary_vector]
            answers = [training_data[z].answer, training_data[z].answer]
            combined_vector = combine_vectors(answers, *vectors)
            combined_distribution.increment_cell(training_data[z].answer, combined_vector)

        # Normalize the combined matrix.
        normalized_combination = NormalizedMatrix(combined_distribution, combine_ids=True)

        # Store both matrices.
        combined_distributions.append(combined_distribution)
        normalized_combined_distributions.append(normalized_combination)

        # Create the final container object:

        ap_svm_dict = dict()  # type: Dict[Tuple[AccessPoint, ...], List[svm_model]]
        ap_svm_dict[access_point_tuple] = primary_svm_list
        ap_svm_dict[aps_being_used] = svms

        combined_features = list()
        half = int(len(features) / 2)
        for f in range(len(features)):
            combined_features.append(features[f] + ap_test_features[f])

        combined_svm = svm_train(test_class[:half], combined_features[:half], '-q')

        p_labs, p_acc, p_vals = svm_predict(y=test_class[half:], x=combined_features[half:], m=combined_svm,
                                            options="-q")
        # print("the accuracy for combined is {}".format(p_acc[0]))
        # Create the object needed:
        final_container = FinalCombinationContainer(
            ap_svm_dict=ap_svm_dict,
            ap_tuples=ap_combination,
            combined_svm=combined_svm,
            normalization=normalized_combination,
            combined_distribution=combined_distribution
        )

        final_combination_container.append(final_container)
        tracker += 1

    return final_combination_container


# def build_combined_distributions_bk(averaged_distributions: Dict[Tuple[AccessPoint, ...],
#                                                               Tuple[NormalizedDistribtuion, List[svm_model]]],
#                                  num_combinations: int,
#                                  training_data: List[Scan]
#                                  ) -> List[FinalCombinationContainer]:
#     # Set combination method:
#     combine_vectors = __weighted_combine_vectors
#
#     # Containers to hold matrices being returned.
#     final_combination_container = list()  # type: List[FinalCombinationContainer]
#     combined_distributions = list()  # type: List[CombinedMatrix]
#     normalized_combined_distributions = list()  # type: List[NormalizedDistribtuion]
#
#     # Containers to hold data from the passed param.
#     access_point_combinations = list(averaged_distributions.keys())
#
#     amount = __nCr(len(averaged_distributions), num_combinations)
#     print("------ There will be {} combinations.".format(int(amount)))
#     tracker = 1
#
#     # For every possible ap combination: # TODO: This should pick the best normalized dist, instead of combining all.
#     for ap_combination in __ap_combinations(access_point_combinations, num_combinations):
#         print("------ Combination number {}:".format(tracker))
#         print("------ Combining {}.".format(ap_combination))
#
#         # Get the normalizations being combined:
#         normalizations = [norm[0] for norm in [averaged_distributions[ap] for ap in ap_combination]]
#
#         # Create an empty combined distribution:
#         combined_distribution = CombinedMatrix(*normalizations)
#         for sample in training_data:
#
#             vectors = list()  # type: List[Dict[int, float]]
#             answers = list()
#
#             for primary_ap in ap_combination:
#                 aps_being_used = [x for x in primary_ap]
#                 features = [x for key, x in sample.scan.items() if key in aps_being_used]
#
#                 matrix = averaged_distributions[primary_ap][0]
#                 svms = averaged_distributions[primary_ap][1]
#
#                 # Get the predicted zone from the svm
#                 zone_predictions = list()
#                 for s in range(len(svms)):
#                     p_labs, p_acc, p_vals = svm_predict(y=[], x=[features], m=svm)
#                     zone_predictions.append(p_labs)
#
#                 best_predict_zone = max(zone_predictions, key=zone_predictions.count)
#                 # Get the predicted zone (the index that matches the highest probability + 1).
#                 predicted_zone = matrix.zones[int(best_predict_zone[0]) - 1]
#                 # Add 1 because the zone = index + 1.
#
#                 # Get the vector from the normalized distribution.
#                 vector = matrix.get_vector(predicted_zone)
#
#                 vectors.append(vector)
#                 answers.append(sample.answer)
#
#             combined_vector = combine_vectors(answers, *vectors)
#             combined_distribution.increment_cell(sample.answer, combined_vector)
#
#         # Normalize the combined matrix.
#         normalized_combination = NormalizedDistribtuion(combined_distribution, combine_ids=True)
#
#         # Store both matrices.
#         combined_distributions.append(combined_distribution)
#         normalized_combined_distributions.append(normalized_combination)
#
#         # Create the final container object:
#
#         ap_svm_dict = dict()  # type: Dict[Tuple[AccessPoint, ...], List[svm_model]]
#
#         for combo in ap_combination:
#             ap_svm_dict[combo] = averaged_distributions[combo][1]
#
#         # Create the object needed:
#         final_container = FinalCombinationContainer(
#             ap_tuples=ap_combination,
#             ap_svm_dict=ap_svm_dict,
#             normalization=normalized_combination,
#             combined_distribution=combined_distribution
#         )
#
#         final_combination_container.append(final_container)
#         tracker += 1
#
#     return final_combination_container


def __sum_vectors(*vectors: Dict[Zone, float]) -> Dict[Zone, float]:
    dic = dict()  # type: Dict[Zone, float]
    for v in vectors:
        for actual_zone, value in v.items():
            if actual_zone not in dic.keys():
                dic[actual_zone] = value
            else:
                dic[actual_zone] += value
    return dic


def __sum_combine_vectors(answers: List[Zone], *vectors: Dict[Zone, float]) -> Dict[Zone, float]:
    vector = __sum_vectors(*vectors)
    length = len(vectors)

    for zone, v in vector.items():
        vector[zone] = v / length

    return vector


def __weighted_combine_vectors(answers: List[Zone], *vectors: Dict[Zone, float]) -> Dict[Zone, float]:
    new_vectors = list()  # type: List[Dict[Zone, float]]
    for answer, vector in zip(answers, vectors):
        alpha = NormalizedMatrix.get_vector_success(answer, vector)
        new_vectors.append(NormalizedMatrix.scalar_vector(vector, alpha))

    return __sum_vectors(*new_vectors)


# JC-01 - add debugging code to dump out the involved vectors
def __dbg_combine_vectors(answers: List[Zone], *vectors: Dict[Zone, float]) -> Dict[Zone, float]:
    new_vectors = list()  # type: List[Dict[Zone, float]]
    for answer, vector in zip(answers, vectors):
        alpha = NormalizedMatrix.get_vector_success(answer, vector)
        new_vectors.append(NormalizedMatrix.scalar_vector(vector, alpha))
    v = __sum_vectors(*new_vectors)

    # if weightcombined work, then return
    z1 = max(v, key=v.get)
    theAnswer = NormalizedMatrix.theAnswer
    if z1 == theAnswer:
        return v

    print("Debug....")
    z1s = " "
    for zone, i in v.items():
        z1s = z1s + str(i) + " ,"

    # not working, want to see what is the difference
    # focus on the best 2
    # make the 2nd best
    v[z1] = 0.0
    z2 = max(v, key=v.get)

    print("the Answer is ", theAnswer, " Weighted Answer is ", z1, " 2nd Zone is ", z2)
    print("Weighted Vector is ", z1s)

    # use the zone weight
    nv = list()
    for vector in vectors:
        alpha = vector[z1]
        nv.append(NormalizedMatrix.scalar_vector(vector, alpha))

    v.clear()
    v = __sum_vectors(*nv)
    z1s = " "
    for zone, i in v.items():
        z1s = z1s + str(i) + " ,"
    print("Weighted Zone 1 Vector is ", z1s)

    nv.clear()
    for vector in vectors:
        alpha = vector[z2]
        nv.append(NormalizedMatrix.scalar_vector(vector, alpha))

    v.clear()
    v = __sum_vectors(*nv)
    z1s = " "
    for zone, i in v.items():
        z1s = z1s + str(i) + " ,"
    print("Weighted Zone 2 Vector is ", z1s)
    print("End Debug....")

    v.clear()
    v = __sum_vectors(*new_vectors)
    return v
