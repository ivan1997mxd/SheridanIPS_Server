from Algorithms.svm.svm import svm_model
from Resources.Objects.Matrices.CombinedDistribution import test_normalized_list, test_svm_matrices
from Resources.Objects.Points.AccessPoint import AccessPoint
from Objects.Collector import IndividualModel
from typing import List, Tuple, Dict
from numpy import average as avg
from sklearn.svm import SVC

from Objects.TestData import TestResult
from Resources.Objects.Matrices.Matrix import Matrix
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix, build_normalized_distribution
from Resources.Objects.Matrices.ProbabilityDistribution import ProbabilityMatrix, build_svm_probability_distributions, \
    build_svm_probability_distribution
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.TestData import Sample
from Resources.Objects.Zone import Zone


class Fold:
    """ Each fold object holds all Probability Distributions built using that Fold's sample data.
    """

    def __init__(self):
        self.__probability_distributions = dict()  # type: Dict[Tuple[AccessPoint, ...], ProbabilityMatrix]
        self.__normalized_distributions = dict()  # type: Dict[Tuple[AccessPoint, ...], NormalizedMatrix]
        self.__test_distributions = dict()  # type: Dict[Tuple[AccessPoint, ...], TestResult]
        self.__trained_models = dict()  # type: Dict[Tuple[AccessPoint, ...], IndividualModel]

    def add_trained_models(self, models: Dict[Tuple[AccessPoint, ...], IndividualModel]):
        self.__trained_models.update(models)

    def add_distributions(self, access_point_tuple: Tuple[AccessPoint, ...], p_dist: ProbabilityMatrix,
                          n_dist: NormalizedMatrix) -> None:
        self.__probability_distributions[access_point_tuple] = p_dist
        self.__normalized_distributions[access_point_tuple] = n_dist

    def create_distributions(self, access_point_combinations: List[Tuple[AccessPoint, ...]],
                             p_list: List[ProbabilityMatrix], n_list: List[NormalizedMatrix],
                             t_list: Dict[NormalizedMatrix, TestResult]):
        for a in range(len(access_point_combinations)):
            ap_tuple = access_point_combinations[a]
            normalized = n_list[a]
            probailied = p_list[a]
            self.__probability_distributions[ap_tuple] = probailied
            self.__normalized_distributions[ap_tuple] = normalized
            self.__test_distributions[ap_tuple] = t_list[normalized]

    @property
    def distributions(self) -> Dict[Tuple[AccessPoint, ...], ProbabilityMatrix]:
        return self.__probability_distributions

    def get_probability_distribution(self, access_point_tuple: Tuple[AccessPoint, ...]) -> ProbabilityMatrix:
        return self.__probability_distributions[access_point_tuple]

    def get_normalized_distribution(self, access_point_tuple: Tuple[AccessPoint, ...]) -> NormalizedMatrix:
        return self.__normalized_distributions[access_point_tuple]

    def get_test_distribution(self, access_point_tuple: Tuple[AccessPoint, ...]) -> TestResult:
        return self.__test_distributions[access_point_tuple]

    def get_best_model(self):
        best_model = list(self.__trained_models.values())[0]
        for ap, model in self.__trained_models.items():
            if best_model.percentage_correct < model.percentage_correct:
                best_model = model
        return best_model

    def get_trained_model(self, access_point_tuple: Tuple[AccessPoint, ...]) -> IndividualModel:
        return self.__trained_models[access_point_tuple]

    def get_SVM(self, access_point_tuple: Tuple[AccessPoint, ...]) -> svm_model:
        return self.get_trained_model(access_point_tuple).svm

    def create_probability_distributions(self):
        for ap_tuple, trained_model in self.__trained_models.items():
            self.__probability_distributions[ap_tuple] = build_svm_probability_distribution(trained_model)

    def create_normalized_distributions(self):
        for ap_tuple, p_dist in self.__probability_distributions.items():
            self.__normalized_distributions[ap_tuple] = build_normalized_distribution(p_dist)

    def create_test_distributions(self,
                                  zones: List[Zone],
                                  testing_class: List[int],
                                  testing_features: List[Dict[AccessPoint, int]]):
        for ap_tuple, n_dist in self.__normalized_distributions.items():
            svm = self.get_SVM(ap_tuple)
            test_result = test_svm_matrices(normalized_distribution=n_dist,
                                            zones=zones,
                                            test_features=testing_features,
                                            test_class=testing_class,
                                            svm=svm)
            self.__test_distributions[ap_tuple] = test_result

    @staticmethod
    def get_average_distribution(access_points: List[AccessPoint],
                                 zones: List[Zone],
                                 distributions: List[NormalizedMatrix]) -> NormalizedMatrix:

        # norm = NormalizedMatrix(Matrix(access_points=access_points, zones=zones, size=len(zones)))
        prob = ProbabilityMatrix(Matrix(access_points=access_points, zones=zones, size=len(zones)))
        # Go cell by cell.
        for i in zones:
            for j in zones:
                # values = list()  # type: # List[float]
                parent_values = list()
                for distribution in distributions:
                    # values.append(distribution.get_value(measured_zone=i, actual_zone=j))
                    parent_values.append(distribution.parent_matrix.get_value(measured_zone=i, actual_zone=j))
                # avg_value = sum(values) / len(values)
                avg_parent_value = sum(parent_values) / len(parent_values)
                prob.set_value(
                    measured_zone=i,
                    actual_zone=j,
                    value=avg_parent_value)
                # norm.parent_matrix.set_value(measured_zone=i, actual_zone=j, value=avg_parent_value)
                # norm.set_value(measured_zone=i, actual_zone=j, value=avg_value)
        normalized_matrix = NormalizedMatrix(prob)
        return normalized_matrix

    def __repr__(self):
        Str = str(self.__trained_models)
        Str += "\nNumber of Distributions: {}".format(len(self.__normalized_distributions))
        return Str

    @staticmethod
    def get_average_test_distribution(distributions: List[TestResult]) -> float:
        test_accuracy = list()
        for distribution in distributions:
            test_accuracy.append(distribution.accuracy)

        return sum(test_accuracy) / len(test_accuracy)


def get_average_rate(fold: Fold):
    return fold.get_normalized_distribution().average_matrix_success
