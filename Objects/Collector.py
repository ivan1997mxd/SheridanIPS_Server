from typing import List, Tuple

from Algorithms.svm.svm import svm_model
from Resources.Objects.Points.AccessPoint import AccessPoint
from sklearn.svm import SVC

from Resources.Objects.Zone import Zone


class IndividualModel:

    def __init__(self,
                 svm: svm_model,
                 access_point_tuple: Tuple[AccessPoint, ...],
                 train_features: List[List[int]],
                 zones: List[Zone],
                 train_classes: List[int],
                 test_features: List[List[int]],
                 test_classes: List[int],
                 predictions: List[int],
                 percentage_correct: float):
        self.__svm = svm
        self.__zones = zones
        self.__access_point_tuple = access_point_tuple
        self.__percentage_correct = percentage_correct
        self.__train_features = train_features
        self.__train_classes = train_classes
        self.__test_features = test_features
        self.__test_classes = test_classes
        self.__predictions = predictions

    @property
    def train_features(self) -> List[List[int]]:
        return self.__train_features

    @property
    def train_classes(self) -> List[int]:
        return self.__train_classes

    @property
    def predictions(self) -> List[int]:
        return self.__predictions

    @property
    def svm(self):
        return self.__svm

    @property
    def percentage_correct(self) -> float:
        return self.__percentage_correct

    @property
    def access_point_tuple(self) -> Tuple[AccessPoint]:
        return self.__access_point_tuple

    @property
    def access_points(self) -> List[AccessPoint]:
        return [*self.__access_point_tuple]

    @property
    def zones(self) -> List[Zone]:
        return self.__zones

    @property
    def test_classes(self):
        return self.__test_classes

    def __repr__(self):
        return "APs: " + str(self.__access_point_tuple) + " Correct: " + str(self.__percentage_correct)


class CollectedModel:
    # TODO: Delete? I don't think this is being used.

    def __init__(self, individual_tuples: List[IndividualModel]):
        self.__individual_tuples = individual_tuples
        self.__average_error = None
        self.__id = None
        self.__best_individual_tuple = None

    @property
    def best_indy(self):
        if self.__best_individual_tuple is None:
            best_tuple = self.__individual_tuples[0]
            for indy in self.__individual_tuples[1:]:
                if indy.percentage_correct > best_tuple.percentage_correct:
                    best_tuple = indy
            self.__best_individual_tuple = best_tuple
            return self.__best_individual_tuple
        return self.__best_individual_tuple

    @property
    def id(self):
        if self.__id is None:
            self.__id = ""
            for ap in self.__individual_tuples[0].access_point_tuple:
                self.__id += " AP: " + str(ap.num)
            return self.__id
        return self.__id

    @property
    def error(self):
        if self.__average_error is None:
            self.__average_error = self.__calculate_error()
            return self.__average_error
        return self.__average_error

    def __calculate_error(self):
        # Error = 1 - (AVG(correct rate))
        correct = 0
        for indy in self.__individual_tuples:
            correct += indy.percentage_correct
        return 1 - (correct / len(self.__individual_tuples))
