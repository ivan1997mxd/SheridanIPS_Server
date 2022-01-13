from itertools import combinations
from time import time

from sklearn.model_selection import KFold
import numpy as np
from math import floor, log
from random import shuffle, choice
from typing import Callable, Dict

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from Algorithms.Combination.Combination import build_combined_distributions, build_jc_combined_distributions, \
    get_combination_function
from Algorithms.svm.svmutil import svm_train, svm_predict
# from Matrices.NormalizedDistribution import sort_matrices
# from libsvm.svmutil import svm_train, svm_predict
from Objects.Collector import *
from Objects.FinalCombinationContainer import get_average_rate
from Objects.Fold import Fold
from Offline.MatrixProduction import create_all_matrices_from_rssi_data, create_all_svm_matrices
from Resources.Objects.Floor import Floor
from Resources.Objects.Matrices.CombinedDistribution import test_combination_matrices, test_svm_matrices
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix, sort_matrices
from Resources.Objects.Matrices.ProbabilityDistribution import __get_access_point_combinations
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Points.AccessPoint import AccessPoint, get_n_ap_combinations, get_ap_combinations
from Resources.Objects.TestData import Sample, create_test_data_list
from Resources.Objects.Zone import Zone
from Resources.Objects.Points.GridPoint_RSSI import GridPoint


def create_compare_combination(num_splits: int,
                               combination_mode: str,
                               location_mode: str,
                               num_combinations: int,
                               mm_ap_dict: dict,
                               data: List[Sample],
                               combine: bool,
                               centroids: List[Centroid],
                               grid_points: List[GridPoint],
                               access_points: List[AccessPoint],
                               zones: List[Zone]):
    jc_time = 0
    gd_time = 0
    start_initialize_time = time()
    # 4. Set K-Fold Splits
    X = np.array(data)
    kf = KFold(n_splits=num_splits)

    # 5. For every fold, for every AP combination, train a new SVM.
    fold_number = 1
    split_kf = kf.split(X)

    all_access_point_combinations = get_ap_combinations(
        access_points=access_points)  # type: List[Tuple[AccessPoint, ...]]
    normalized_list = dict()  # type: Dict[int, List[NormalizedMatrix]]
    folds = dict()  # type: Dict[int, Fold]
    best_ap_list = dict()  # type: Dict[int, List[Tuple[AccessPoint, ...]]]
    accuracy_check = dict()  # type: Dict[Tuple[AccessPoint, ...], List[svm_model]]
    best_gd_models = dict()  # type: Dict[Tuple[AccessPoint, ...], Tuple[NormalizedMatrix, List[svm_model]]]
    best_jc_models = dict()  # type: Dict[Tuple[AccessPoint, ...], Tuple[NormalizedMatrix, Dict[Tuple[AccessPoint, ...], List[svm_model]]]]
    ig_models = list()
    end_initialize_time = time()
    initialize_time = end_initialize_time - start_initialize_time
    # print("initialize time: {}s.".format(initialize_time))
    jc_time += initialize_time
    gd_time += initialize_time
    for train_indices, test_indices in split_kf:

        start_data_time = time()
        print("Starting Fold {} with {}.".format(fold_number, location_mode))

        fold = Fold()

        train_samples = list()  # type: List[Sample]
        test_samples = list()  # type: List[Sample]
        train_features = list()  # type: List[Dict[AccessPoint, int]]
        train_classes = list()  # type: List[int]
        test_features = list()  # type: List[Dict[AccessPoint, int]]
        test_classes = list()  # type: List[int]

        # change to NNv4 mode
        for num in train_indices:
            train_samples.append(data[num])
            train_features.append(data[num].scan)
            train_classes.append(data[num].answer.num)

        for num in test_indices:
            test_samples.append(data[num])
            test_features.append(data[num].scan)
            test_classes.append(data[num].answer.num)

        # info gain selection
        ig_model = choose_best_info_gain(train_samples, access_points)
        ig_models.append(ig_model)
        d = 2
        trained_models = list()  # type: List[IndividualModel]
        end_data_time = time()
        data_time = end_data_time - start_data_time
        # print("data time: {}s.".format(data_time))
        jc_time += data_time
        gd_time += data_time
        while d < len(access_points) + 1:
            start_combination_time = time()
            access_point_combinations = get_n_ap_combinations(access_points, d)  # type: List[Tuple[AccessPoint, ...]]
            end_combination_time = time()
            combination_time = end_combination_time - start_combination_time
            # print("combination time: {}s.".format(combination_time))
            jc_time += combination_time
            gd_time += combination_time
            # 6. Get all AP Combinations
            S_score = dict()  # type: Dict[Tuple[AccessPoint, ...], float]

            # ap_features = list()
            #
            # for feature_set in train_features:
            #     ap_features.append(
            #         [value for key, value in feature_set.items()])
            #
            # param_grid = {"gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 0.25, 0.5, 0.75, 0.3, 0.2, 0.15, 1000],
            #               "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            # grid_search = GridSearchCV(SVC(decision_function_shape='ovo'), param_grid, cv=5)
            # grid_search.fit(ap_features, train_classes)
            # print("Best parameters:{}".format(grid_search.best_params_))
            # gamma = grid_search.best_params_['gamma']
            # c = grid_search.best_params_['C']
            start_svm_time = time()
            for access_point_tuple in access_point_combinations:
                # print("--- Working on AP Combination: {}".format(access_point_tuple))
                ap_train_features = list()
                ap_test_features = list()
                aps_being_used = [x for x in access_point_tuple]
                for feature_set in train_features:
                    ap_train_features.append(
                        [value for key, value in feature_set.items() if key in aps_being_used])

                for feature_set in test_features:
                    ap_test_features.append([value for key, value in feature_set.items() if key in aps_being_used])

                # m = svm_train(train_classes, ap_train_features, '-q -c {} -g {}'.format(c, gamma))
                m = svm_train(train_classes, ap_train_features, '-q')
                margins = m.get_labels()
                score = 0
                for margin in margins:
                    score += 0.5 * margin
                S_score[access_point_tuple] = score
                p_labs, p_acc, p_vals = svm_predict(y=test_classes, x=ap_test_features, m=m, options='-q')
                if access_point_tuple in accuracy_check:
                    accuracy_check[access_point_tuple].append(m)
                else:
                    accuracy_check[access_point_tuple] = [m]

                individual_model = IndividualModel(svm=m,
                                                   access_point_tuple=access_point_tuple,
                                                   zones=zones,
                                                   train_features=ap_train_features,
                                                   train_classes=train_classes,
                                                   test_features=ap_test_features,
                                                   test_classes=test_classes,
                                                   predictions=p_labs,
                                                   percentage_correct=p_acc[0])
                fold.add_trained_models({access_point_tuple: individual_model})
                trained_models.append(individual_model)
            end_svm_time = time()
            svm_time = end_svm_time - start_svm_time
            # print("svm time: {}s.".format(svm_time))
            gd_time += svm_time
            if location_mode == "SVM":
                jc_time += svm_time
            # results = create_all_svm_matrices(trained_models=trained_models)
            # normalized_list[d] = results[1]
            start_find_time = time()
            best_one = tuple()  # type:Tuple[AccessPoint, ...]
            min_value = min(S_score.values())
            for keys, values in S_score.items():
                if values == min_value:
                    best_one = keys
            if d in best_ap_list.keys():
                best_ap_list[d].append(best_one)
            else:
                best_ap_list[d] = [best_one]
            end_find_time = time()
            find_time = end_find_time - start_find_time
            # print("find time: {}s.".format(find_time))
            gd_time += find_time
            # print("finish GD training for d = {}, best ap combination is {}".format(d, best_one))
            if location_mode != "SVM":
                start_matrix_time = time()
                distributions = create_all_matrices_from_rssi_data(
                    access_points=access_points,
                    access_point_combinations=access_point_combinations,
                    centroids=centroids,
                    grid_points=grid_points,
                    zones=zones,
                    training_data=train_samples,
                    testing_data=test_samples,
                    combination_mode=combination_mode,
                    location_mode=location_mode,
                    num_combinations=num_combinations,
                    do_combination=False)
                # print("First Run")
                probability_distributions = distributions[0]
                normalized_distributions = distributions[1]
                fold.create_distributions(access_point_combinations=access_point_combinations,
                                          p_list=probability_distributions,
                                          n_list=normalized_distributions)
                end_matrix_time = time()
                matrix_time = end_matrix_time - start_matrix_time
                # print("matrix time: {}s.".format(matrix_time))
                jc_time += matrix_time
            d += 1
        folds[fold_number] = fold
        print("Completed Fold {}.".format(fold_number))
        fold_number += 1
    if location_mode == "SVM":
        start_distributions_time = time()
        print("Creating Probability and Normalized Probability Distributions...")
        for fold in folds.values():
            fold.create_probability_distributions()
            fold.create_normalized_distributions()
        print("Completed. There are {} Probability and Normalized Distributions.".format(len(folds)))
        end_distributions_time = time()
        distributions_time = end_distributions_time - start_distributions_time
        print("distribution time: {}s.".format(distributions_time))
        jc_time += distributions_time
        gd_time += distributions_time
    print("Averaging the matrices produced from Folding...")
    averaged_normalized_distributions = dict()
    for ap_tuple in all_access_point_combinations:
        # print("--- Working on AP Combination: {}.".format(ap_tuple))
        start_average_time = time()
        distributions_to_average = list()  # type: List[NormalizedMatrix]
        svms_to_store = list()  # type: List[svm_model]
        for fold in folds.values():
            distributions_to_average.append(fold.get_normalized_distribution(ap_tuple))
            svms_to_store.append(fold.get_SVM(ap_tuple))
        averaged_normalized_distribution = Fold.get_average_distribution(access_points=[*ap_tuple],
                                                                         zones=zones,
                                                                         distributions=distributions_to_average)
        if location_mode == "SVM":
            averaged_normalized_distributions[ap_tuple] = averaged_normalized_distribution, svms_to_store
            # print("--- Completed. {} Normalizations, and {} SVMs have been stored."
            #       .format(len(distributions_to_average), len(svms_to_store)))
        else:
            averaged_normalized_distributions[ap_tuple] = averaged_normalized_distribution
        end_average_time = time()
        average_time = end_average_time - start_average_time
        # print("Average distribution time: {}s.".format(average_time))
        jc_time += average_time
    print("Completed. There are {} Normalized Distributions ready for combination."
          .format(len(averaged_normalized_distributions.values())))

    best_ig_model = list()
    for i in range(len(access_points)):
        ap_list = [model[i] for model in ig_models]
        best_ig_model.append(max(ap_list, key=ap_list.count))

    if location_mode == "SVM":
        # Build Combined Distributions.
        # final_containers = build_jc_combined_distributions(
        #     averaged_distributions=averaged_normalized_distributions,
        #     num_combinations=num_combinations,
        #     training_data=data)

        # print("--- Completed Combining All {} Matrices.".format(num_combinations))
        # sorted_final = sorted(final_containers, key=get_average_rate, reverse=True)
        # combined_matrix = [matrix.normalization for matrix in sorted_final]
        # normalized_matrix = [value[0] for value in averaged_normalized_distributions.values()]

        start_jc_model_time = time()
        for d in range(2, len(access_points) + 1):
            dict_best_model = dict()
            average_list = [value[0] for key, value in averaged_normalized_distributions.items() if len(key) == d]
            sort_matrices(average_list)
            avg_ap_tuple = tuple(average_list[0].access_points)
            dict_best_model[avg_ap_tuple] = averaged_normalized_distributions[avg_ap_tuple][1]
            best_jc_models[tuple(avg_ap_tuple)] = averaged_normalized_distributions[avg_ap_tuple][0], dict_best_model
        # best_jc_models[sorted_final[0].ap_tuples] = sorted_final[0].normalization, sorted_final[0].ap_svm_dict
        end_jc_model_time = time()
        jc_model_time = end_jc_model_time - start_jc_model_time
        jc_time += jc_model_time
        start_gd_model_time = time()
        for index, value in best_ap_list.items():
            ap_set = max(set(value), key=value.count)
            best_matrix = averaged_normalized_distributions[ap_set][0]
            best_d_model = accuracy_check[ap_set]
            best_gd_models[ap_set] = best_matrix, best_d_model
            # print("d = {}, the Best AP Combination is: {}".format(index, ap_set))
        end_gd_model_time = time()
        gd_model_time = end_gd_model_time - start_gd_model_time
        gd_time += gd_model_time
        print("JC Time is {}, GD time is {} when location mode is SVM".format(jc_time, gd_time))
        return best_jc_models, best_gd_models, jc_time, gd_time, accuracy_check, best_ig_model
    else:
        start_jc_model_time = time()

        averaged_normalized_list = sorted(averaged_normalized_distributions.values(),
                                          key=lambda x: x.average_matrix_success, reverse=True)
        for i in range(2, len(access_points) + 1):
            best_normalization = next(normalized_dist for normalized_dist in averaged_normalized_list if
                                      len(normalized_dist.access_points) == i)
            jc_ap_tuple = best_normalization.access_points_tuple
            best_jc_models[jc_ap_tuple] = best_normalization, {jc_ap_tuple: accuracy_check[jc_ap_tuple]}
        if combine and len(access_points) >= 4:
            list_svm_model = list()
            best_combined_matrixs = list()

            mm_normalized_list = [normalized_matrix for normalized_matrix in
                                  list(averaged_normalized_distributions.values()) if
                                  normalized_matrix.access_points in list(mm_ap_dict.values())]
            sort_matrices(mm_normalized_list)
            for num_combinations in range(2, 4):
                combined_distributions, normalized_combined_distributions = build_combined_distributions(
                    normalized_distributions=mm_normalized_list,
                    training_data=data,
                    centroids=centroids,
                    grid_points=grid_points,
                    zones=zones,
                    num_combination=num_combinations,
                    combination_mode=combination_mode,
                    location_mode=location_mode
                )
                sort_matrices(normalized_combined_distributions)
                best_combined_matrix = normalized_combined_distributions[0]
                best_combined_aps = best_combined_matrix.access_points_combo
                best_combined_matrixs.append(best_combined_matrix)
                best_svm_dict = dict()
                for aps in best_combined_aps:
                    best_svm_dict[aps] = accuracy_check[aps]
                list_svm_model.append(best_svm_dict)
            for d in range(len(best_combined_matrixs)):
                best_combined_matrix = best_combined_matrixs[d]
                best_svm_dict = list_svm_model[d]
                best_jc_models[best_combined_matrix.access_points_tuple] = best_combined_matrix, best_svm_dict
        end_jc_model_time = time()
        jc_model_time = end_jc_model_time - start_jc_model_time
        jc_time += jc_model_time
        start_gd_model_time = time()
        for index, value in best_ap_list.items():
            ap_set = max(set(value), key=value.count)
            best_matrix = next(normalized_dist for normalized_dist in averaged_normalized_list if
                               normalized_dist.access_points_tuple == ap_set)
            best_d_model = accuracy_check[ap_set]
            best_gd_models[ap_set] = best_matrix, best_d_model
            print("d = {}, the Best AP Combination is: {}".format(index, ap_set))
        # combination_method = get_combination_function(combination_mode)
        # test_results = test_combination_matrices(normalized_combined_distributions=normalized_combined_distributions,
        #                                          centroids=centroids,
        #                                          zones=zones,
        #                                          testing_data=data,
        #                                          combination_method=combination_method)

        # print("-- Completed Testing Distributions.")
        end_gd_model_time = time()
        gd_model_time = end_gd_model_time - start_gd_model_time
        gd_time += gd_model_time
        print("JC Time is {}, GD time is {} when location mode is not SVM".format(jc_time, gd_time))
        return best_jc_models, best_gd_models, jc_time, gd_time, accuracy_check, best_ig_model


def gd_approach(num_splits: int,
                data: List[Sample],
                access_points: List[AccessPoint]):
    # 4. Set K-Fold Splits
    X = np.array(data)
    kf = KFold(n_splits=num_splits)

    # 5. For every fold, for every AP combination, train a new SVM.
    fold_number = 1
    split_kf = kf.split(X)

    best_ap_set = dict()  # type: Dict[int, Tuple[AccessPoint, ...]]
    best_ap_list = dict()  # type: Dict[int, List[Tuple[AccessPoint, ...]]]

    for train_indices, test_indices in split_kf:
        print("Starting Fold {}.".format(fold_number))

        train_features = list()  # type: List[Dict[AccessPoint, int]]
        train_classes = list()  # type: List[int]
        test_features = list()  # type: List[Dict[AccessPoint, int]]
        test_classes = list()  # type: List[int]

        # change to NNv4 mode
        for num in train_indices:
            train_features.append(data[num].scan)
            train_classes.append(data[num].answer.num)

        for num in test_indices:
            test_features.append(data[num].scan)
            test_classes.append(data[num].answer.num)

        d = 2
        while d < len(access_points) + 1:
            access_point_combinations = get_n_ap_combinations(access_points, d)  # type: List[Tuple[AccessPoint, ...]]
            # 6. Get all AP Combinations
            S_score = dict()  # type: Dict[Tuple[AccessPoint, ...], float]
            accuracy_check = dict()  # type: Dict[Tuple[AccessPoint, ...], float]
            ap_features = list()

            for feature_set in train_features:
                ap_features.append(
                    [value for key, value in feature_set.items()])
            param_grid = {"gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 0.25, 0.5, 0.75, 0.3, 0.2, 0.15, 1000],
                          "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            grid_search = GridSearchCV(SVC(decision_function_shape='ovo'), param_grid, cv=5)
            grid_search.fit(ap_features, train_classes)
            print("Best parameters:{}".format(grid_search.best_params_))
            gamma = grid_search.best_params_['gamma']
            c = grid_search.best_params_['C']

            for access_point_tuple in access_point_combinations:
                # print("--- Working on AP Combination: {}".format(access_point_tuple))
                ap_train_features = list()
                ap_test_features = list()

                aps_being_used = [x for x in access_point_tuple]
                for feature_set in train_features:
                    ap_train_features.append(
                        [value for key, value in feature_set.items() if key in aps_being_used])

                for feature_set in test_features:
                    ap_test_features.append([value for key, value in feature_set.items() if key in aps_being_used])

                m = svm_train(train_classes, ap_train_features, '-q -c {} -g {}'.format(c, gamma))
                margins = m.get_labels()
                score = 0
                for margin in margins:
                    score += 0.5 * margin
                S_score[access_point_tuple] = score
                p_labs, p_acc, p_vals = svm_predict(y=test_classes, x=ap_test_features, m=m)
                accuracy_check[access_point_tuple] = p_acc[0]

            best_one = tuple()  # type:Tuple[AccessPoint, ...]
            best_accuracy = tuple()  # type:Tuple[AccessPoint, ...]
            min_value = min(S_score.values())
            max_value = max(accuracy_check.values())
            for key, value in accuracy_check.items():
                if value == max_value:
                    best_accuracy = key
            for keys, values in S_score.items():
                if values == min_value:
                    best_one = keys
            # best_accuracy = accuracy_check[best_one]
            if d in best_ap_list.keys():
                best_ap_list[d].append(best_one)
            else:
                best_ap_list[d] = [best_one]
            print("finish GD training for d = {}, best ap combination is {}, best accuracy is {}".format(d, best_one,
                                                                                                         best_accuracy))
            d += 1
        print("Completed Fold {}.".format(fold_number))
        fold_number += 1
    for index, value in best_ap_list.items():
        ap_set = max(set(value), key=value.count)
        best_ap_set[index] = ap_set
        print("d = {}, the Best AP Combination is: {}".format(index, ap_set))
    return best_ap_set


def average_normalized_distribution(folds: Dict[int, Dict[int, List[NormalizedMatrix]]],
                                    access_points: List[AccessPoint], zones: List[Zone]):
    average_normalized_all = dict()  # type: Dict[int, List[NormalizedMatrix]]
    d = 2
    while d < len(access_points) + 1:
        averaged_normalized_distributions = list()  # type: List[NormalizedMatrix]
        for access_point_tuple in get_n_ap_combinations(access_points, d):
            # print("--- Working on AP Combination: {}.".format(access_point_tuple))
            distributions_to_average = list()  # type: List[NormalizedMatrix]
            access_list = [*access_point_tuple]
            for distribution in folds.values():
                target = distribution[d]
                for value in target:
                    if access_list == value.access_points:
                        distributions_to_average.append(value)

            averaged_normalized_distribution = Fold.get_average_distribution(access_points=access_list,
                                                                             zones=zones,
                                                                             distributions=distributions_to_average)
            averaged_normalized_distributions.append(averaged_normalized_distribution)
            print("--- Completed. average {} Normalizations."
                  .format(len(distributions_to_average)))
        sort_matrices(averaged_normalized_distributions)
        average_normalized_all[d] = averaged_normalized_distributions
        d += 1
    return average_normalized_all


def gd_train(training_data: List[Sample], num_splits: int,
             access_points: List[AccessPoint]):
    X = np.array(training_data)
    kf = KFold(n_splits=num_splits)
    # folds = dict()  # type: Dict[int, Fold]
    fold_number = 1
    best_ap_set = dict()  # type: Dict[int, Tuple[AccessPoint, ...]]
    best_ap_list = dict()  # type: Dict[int, List[Tuple[AccessPoint, ...]]]
    for train_indices, test_indices in kf.split(X):
        print("Starting Fold {}.".format(fold_number))

        train_features = list()  # type: List[Dict[AccessPoint, int]]
        train_classes = list()  # type: List[Zone]
        test_features = list()  # type: List[Dict[AccessPoint, int]]
        test_classes = list()  # type: List[Zone]

        for num in train_indices:
            train_features.append(training_data[num].scan)
            train_classes.append(training_data[num].answer)

        for num in test_indices:
            test_features.append(training_data[num].scan)
            test_classes.append(training_data[num].answer)

        d = 2
        while d < len(access_points):
            access_point_combinations = get_n_ap_combinations(access_points, d)  # type: List[Tuple[AccessPoint, ...]]
            # 6. Get all AP Combinations
            S_score = dict()  # type: Dict[Tuple[AccessPoint, ...], float]
            for access_point_tuple in access_point_combinations:
                print("--- Working on AP Combination: {}".format(access_point_tuple))
                ap_train_features = list()
                ap_test_features = list()

                aps_being_used = [x.num for x in access_point_tuple]
                for feature_set in train_features:
                    ap_train_features.append(
                        [x for index, x in enumerate(feature_set) if (index + 1) in aps_being_used])

                for feature_set in test_features:
                    ap_test_features.append([x for index, x in enumerate(feature_set) if (index + 1) in aps_being_used])

                print("---------------------------------------------\n")
                m = svm_train(train_classes, ap_train_features)
                margins = m.get_labels()
                score = 0
                for margin in margins:
                    score += 0.5 * margin
                S_score[access_point_tuple] = score
                # p_labs, p_acc, p_vals = svm_predict(y=test_classes, x=ap_test_features, m=m)
            best_one = tuple()  # type:Tuple[AccessPoint, ...]
            min_value = min(S_score.values())
            print(min_value)
            for keys, values in S_score.items():
                if values == min_value:
                    best_one = keys
            if d in best_ap_list.keys():
                best_ap_list[d].append(best_one)
            else:
                best_ap_list[d] = [best_one]
            print("finish GD training for d = {}, best ap combination is {}".format(d, best_one))
            d += 1
        print("Completed Fold {}.".format(fold_number))
        fold_number += 1
    for index, value in best_ap_list.items():
        ap_set = max(set(value), key=value.count)
        best_ap_set[index] = ap_set
        print("d = {}, the Best AP Combination is: {}".format(index, ap_set))

    return best_ap_set


def create_kflod_combination(train_mode: str,
                             selection_mode: str,
                             num_combination: int,
                             centroids: List[Centroid],
                             grid_points: List[GridPoint],
                             access_points: List[AccessPoint],
                             zones: List[Zone],
                             num_splits: int,
                             training_data: List[Sample]):
    # Initialize the k-fold
    X = np.array(training_data)
    kf = KFold(n_splits=num_splits)

    fold_number = 1
    split_kf = kf.split(X)
    folds = dict()  # type: Dict[int, Fold]
    best_ap_list = dict()  # type: Dict[int, List[Tuple[AccessPoint, ...]]]
    accuracy_check = dict()  # type: Dict[Tuple[AccessPoint, ...], List[svm_model]]
    best_gd_models = dict()  # type: Dict[Tuple[AccessPoint, ...], Tuple[NormalizedMatrix, List[svm_model]]]
    best_jc_models = dict()  # type: Dict[Tuple[AccessPoint, ...], Tuple[NormalizedMatrix, List[svm_model]]]
    all_access_point_combinations = get_ap_combinations(access_points=access_points)
    for train_indices, test_indices in split_kf:
        print("Starting Fold {} with {}.".format(fold_number, train_mode))

        fold = Fold()
        train_samples = list()  # type: List[Sample]
        test_samples = list()  # type: List[Sample]
        train_features = list()  # type: List[Dict[AccessPoint, int]]
        train_classes = list()  # type: List[int]
        test_features = list()  # type: List[Dict[AccessPoint, int]]
        test_classes = list()  # type: List[int]

        # prepare data for train and test
        for num in train_indices:
            train_samples.append(training_data[num])
            train_features.append(training_data[num].scan)
            train_classes.append(training_data[num].answer.num)

        for num in test_indices:
            test_samples.append(training_data[num])
            test_features.append(training_data[num].scan)
            test_classes.append(training_data[num].answer.num)

        # Train SVM model
        d = 3
        trained_models = list()  # type: List[IndividualModel]
        while d < len(access_points) + 1:
            # Get list of ap combinations
            access_point_combinations = get_n_ap_combinations(access_points, d)  # type: List[Tuple[AccessPoint, ...]]
            # Get dict for store score
            score_dict = dict()  # type: Dict[Tuple[AccessPoint, ...], float]
            for access_point_tuple in access_point_combinations:
                # for every ap tuple, get the RSSIs
                ap_train_features = list()
                ap_test_features = list()
                aps_being_used = [x for x in access_point_tuple]
                for feature_set in train_features:
                    ap_train_features.append(
                        [value for key, value in feature_set.items() if key in aps_being_used])

                for feature_set in test_features:
                    ap_test_features.append([value for key, value in feature_set.items() if key in aps_being_used])
                # Train svm model
                m = svm_train(train_classes, ap_train_features, '-q')
                margins = m.get_labels()
                score = 0
                for margin in margins:
                    score += 0.5 * margin
                score_dict[access_point_tuple] = score
                p_labs, p_acc, p_vals = svm_predict(y=test_classes, x=ap_test_features, m=m, options='-q')
                if access_point_tuple in accuracy_check:
                    accuracy_check[access_point_tuple].append(m)
                else:
                    accuracy_check[access_point_tuple] = [m]
                individual_model = IndividualModel(svm=m,
                                                   access_point_tuple=access_point_tuple,
                                                   zones=zones,
                                                   train_features=ap_train_features,
                                                   train_classes=train_classes,
                                                   test_features=ap_test_features,
                                                   test_classes=test_classes,
                                                   predictions=p_labs,
                                                   percentage_correct=p_acc[0])
                fold.add_trained_models({access_point_tuple: individual_model})
                trained_models.append(individual_model)
            best_one = tuple()  # type:Tuple[AccessPoint, ...]
            min_value = min(score_dict.values())
            for keys, values in score_dict.items():
                if values == min_value:
                    best_one = keys
            if d in best_ap_list.keys():
                best_ap_list[d].append(best_one)
            else:
                best_ap_list[d] = [best_one]
            if train_mode != "SVM":
                distributions = create_all_matrices_from_rssi_data(
                    access_points=access_points,
                    access_point_combinations=access_point_combinations,
                    centroids=centroids,
                    grid_points=grid_points,
                    zones=zones,
                    training_data=train_samples,
                    testing_data=test_samples,
                    combination_mode="WGT",
                    location_mode=train_mode,
                    num_combinations=num_combination,
                    do_combination=False)

                probability_distributions = distributions[0]
                normalized_distributions = distributions[1]
                fold.create_distributions(access_point_combinations=access_point_combinations,
                                          p_list=probability_distributions,
                                          n_list=normalized_distributions)
            d += 1
        folds[fold_number] = fold
        print("Completed Fold {}.".format(fold_number))
        fold_number += 1

    if train_mode == "SVM":
        for fold in folds.values():
            fold.create_probability_distributions()
            fold.create_normalized_distributions()
        print("Completed. There are {} Probability and Normalized Distributions.".format(len(folds)))
    print("Averaging the matrices produced from Folding...")
    averaged_normalized_distributions = dict()
    for ap_tuple in all_access_point_combinations:
        distributions_to_average = list()  # type: List[NormalizedMatrix]
        svms_to_store = list()  # type: List[svm_model]
        for fold in folds.values():
            distributions_to_average.append(fold.get_normalized_distribution(ap_tuple))
            svms_to_store.append(fold.get_SVM(ap_tuple))
        # Get the average normalized distribution:
        averaged_normalized_distribution = Fold.get_average_distribution(access_points=[*ap_tuple],
                                                                         zones=zones,
                                                                         distributions=distributions_to_average)
        if selection_mode == "GD":
            averaged_normalized_distributions[ap_tuple] = averaged_normalized_distribution, svms_to_store
        else:
            averaged_normalized_distributions[ap_tuple] = averaged_normalized_distribution
    print("Completed. There are {} Normalized Distributions ready for combination."
          .format(len(averaged_normalized_distributions.values())))

    if selection_mode == "JC":
        averaged_normalized_list = sorted(averaged_normalized_distributions.values(),
                                          key=lambda x: x.average_matrix_success, reverse=True)
        for i in range(3, len(access_points) + 1):
            best_normalization = next(normalized_dist for normalized_dist in averaged_normalized_list if len(normalized_dist.access_points) == i)
            jc_ap_tuple = best_normalization.access_points_tuple
            best_jc_models[jc_ap_tuple] = best_normalization, accuracy_check[jc_ap_tuple]
            print("JC select:{}".format(jc_ap_tuple))
        return best_jc_models
    else:
        for index, value in best_ap_list.items():
            ap_set = max(set(value), key=value.count)
            best_matrix = averaged_normalized_distributions[ap_set][0]
            best_d_model = accuracy_check[ap_set]
            best_gd_models[ap_set] = best_matrix, best_d_model
        return best_gd_models


def calc_shannon_ent(training_data: List[Sample]):
    num_entries = len(training_data)  # 返回数据集的行数
    zone_counts = {}  # 保存每个标签(Label)出现次数的字典
    for data in training_data:  # 对每组特征向量进行统计
        current_zone = data.answer  # 提取标签(Label)信息
        if current_zone not in zone_counts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            zone_counts[current_zone] = 0
        zone_counts[current_zone] += 1  # Label计数
    shannon_ent = 0.0  # 经验熵(香农熵)
    for key in zone_counts:  # 计算香农熵
        prob = float(zone_counts[key]) / num_entries  # 选择该标签(Label)的概率
        shannon_ent -= prob * log(prob, 2)  # 2 or e
    return shannon_ent  # 返回经验熵(香农熵)


def split_data_set(training_data: List[Sample], ap: AccessPoint, value: int):
    ret_data_list = []  # 创建返回的数据集列表
    for data in training_data:  # 遍历数据集
        ap_set = data.scan
        if ap in ap_set:
            if ap_set[ap] == value:
                new_set = {i: ap_set[i] for i in ap_set if i != ap}
                ret_data_list.append(Sample(data.answer, new_set))  # 去掉axis特征
        else:
            ret_data_list.append(Sample(data.answer, ap_set))
    return ret_data_list  # 返回划分后的数据集


def choose_best_info_gain(training_data: List[Sample], access_points: List[AccessPoint]):
    info_gain_dict = dict()
    base_entropy = calc_shannon_ent(training_data)  # 计算数据集的香农熵
    print("base entropy is {}".format(base_entropy))
    for ap in access_points:  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        rssi_list = [data.scan[ap] for data in training_data if ap in data.scan]
        unique_rssi = set(rssi_list)  # 创建set集合{},元素不可重复
        new_entropy = 0.0  # 经验条件熵
        for value in unique_rssi:  # 计算信息增益
            sub_data_set = split_data_set(training_data, ap, value)  # subDataSet划分后的子集
            prob = len(sub_data_set) / float(len(training_data))  # 计算子集的概率
            new_entropy += prob * calc_shannon_ent(sub_data_set)  # 根据公式计算经验条件熵
        info_gain = base_entropy - new_entropy  # 信息增益
        info_gain_dict[ap] = info_gain
        print("AP:%s:%.3f" % (ap.num, info_gain))  # 打印每个特征的信息增益

    return sorted(info_gain_dict.keys(),
                  key=info_gain_dict.get, reverse=True)

