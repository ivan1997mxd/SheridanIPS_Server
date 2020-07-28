from typing import Tuple, List, Dict

from sklearn.svm import SVC

# from Matrices.CombinedDistribution import CombinedMatrix
from Algorithms.svm.svm import svm_model
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix
from Resources.Objects.Points.AccessPoint import AccessPoint


class FinalCombinationContainer:
    """ Needed a container object to hold all associated data.

        This object holds a final combination matrix. It also holds all the SVMs, Averagd Matrix, and AP Tuples
        used to generate the final combination matrix.
    """

    def __init__(self,
                 ap_tuples: List[Tuple[AccessPoint, ...]],
                 ap_svm_dict: Dict[Tuple[AccessPoint, ...], List[svm_model]],
                 combined_svm: svm_model,
                 normalization: NormalizedMatrix,
                 combined_distribution):
        self.__ap_tuples = ap_tuples  # type: List[Tuple[AccessPoint, ...]]
        self.__normalization = normalization  # type: NormalizedMatrix
        self.__combined_svm = combined_svm
        self.__ap_svm_dict = ap_svm_dict
        self.__combined_distribution = combined_distribution

    @property
    def combination(self):
        return self.__combined_distribution

    @property
    def normalization(self) -> NormalizedMatrix:
        return self.__normalization

    @property
    def ap_list(self) -> List[AccessPoint]:
        ap_list = list()  # type: List[AccessPoint]
        for ap_tuple in self.__ap_tuples:
            for ap in ap_tuple:
                ap_list.append(ap)
        return ap_list

    @property
    def ap_tuples(self) -> Tuple[Tuple[AccessPoint, ...]]:
        return tuple(self.__ap_tuples)

    @property
    def combined_svm(self) -> svm_model:
        return self.__combined_svm

    @property
    def ap_svm_dict(self) -> Dict[Tuple[AccessPoint, ...], List[svm_model]]:
        return self.__ap_svm_dict

    def svm_list(self, ap_tuple: Tuple[AccessPoint, ...]) -> List[svm_model]:
        return self.__ap_svm_dict[ap_tuple]


def get_average_rate(finalCombinationContainer: FinalCombinationContainer):
    return finalCombinationContainer.normalization.average_matrix_success
