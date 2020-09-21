from Algorithms.svm.svm import svm_model
from Algorithms.svm.svmutil import svm_predict
from Objects.FinalCombinationContainer import FinalCombinationContainer
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix
from Resources.Objects.Matrices.Matrix import Matrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.TestData import Sample, TestResult
from Resources.Objects.Points.Centroid import Centroid
from Algorithms.NearestNeighbour.NNv4 import get_NNv4, get_NNv4_RSSI
from Resources.Objects.Zone import Zone, get_zone
from typing import List, Dict, Callable, Tuple


class CombinedMatrix(Matrix):
    # region Constructor
    def __init__(self, *normalizations: NormalizedMatrix, size: int):
        self.__normalizations = [*normalizations]  # type: List[NormalizedMatrix]
        self.__access_points = list()  # type: List[AccessPoint]
        self.__zones = list()  # type: List[Zone]
        self.__id = None  # type: Union[None, str]

        for norm in self.__normalizations:
            for ap in norm.access_points:
                self.__access_points.append(ap)
            for zone in norm.zones:
                if zone not in self.__zones:
                    self.__zones.append(zone)

        super(CombinedMatrix, self).__init__(access_points=self.__access_points, zones=self.__zones, size=size)

    # endregion

    # region Properties
    @property
    def normalizations(self) -> List[NormalizedMatrix]:
        return self.__normalizations

    # endregion

    # region Methods
    def increment_cell(self, actual_zone: Zone, resultant_vector: Dict[Zone, float]) -> None:
        # Find the highest probability, and check if it occurs more than once:
        highest_probability = max(resultant_vector.values())
        num_occurrences = 0

        for prob in resultant_vector.values():
            if prob == highest_probability:
                num_occurrences += 1

        # Find the zones being reported by the highest probabilities, and increment them:
        for zone, value in resultant_vector.items():
            if value == highest_probability:
                self.increment_value(zone, actual_zone, 1 / num_occurrences)

    # endregion

    # region Override Methods
    @property
    def id(self) -> str:
        if self.__id is not None:
            return self.__id

        Str = ""
        for norm in self.__normalizations:
            aps = ""
            for ap in norm.access_points:
                aps += str(ap.num)
            Str += aps + " U "
        self.__id = Str[:-3]
        return self.__id
    # endregion


def test_normalized_list(normalized_list: List[NormalizedMatrix],
                         centroids: List[Centroid],
                         zones: List[Zone],
                         testing_data: List[Sample]):
    test_results = dict()  # type: Dict[NormalizedMatrix, TestResult]
    for normalized in normalized_list:
        test_result = TestResult()

        # JC-01 used the measured/reported zone instead of the actual zone to the algorithm.
        for sample in testing_data:
            ap_rssi_dict = sample.get_ap_rssi_dict(*normalized.access_points)
            coord = get_NNv4_RSSI(centroid_points=centroids, rssis=ap_rssi_dict)
            zone = get_zone(zones=zones, co_ordinate=coord)

            vector = normalized.get_vector(zone)
            test_result.record(sample.answer, vector)

        test_results[normalized] = test_result

    return test_results


def test_normalized_dict(normalized_dict: Dict[int, List[NormalizedMatrix]],
                         centroids: List[Centroid],
                         zones: List[Zone],
                         testing_data: List[Sample]
                         ) -> Dict[int, List[TestResult]]:
    test_dict = dict()  # type: Dict[int, List[TestResult]]
    for d, normalized_matrices in normalized_dict.items():
        test_results = list()  # type: List[TestResult]
        for normalized_matrix in normalized_matrices:
            test_result = TestResult()
            for sample in testing_data:
                ap_rssi_dict = sample.get_ap_rssi_dict(*normalized_matrix.access_points)
                coord = get_NNv4_RSSI(centroid_points=centroids, rssis=ap_rssi_dict)
                zone = get_zone(zones=zones, co_ordinate=coord)

                vector = normalized_matrix.get_vector(zone)
                test_result.record(sample.answer, vector)
            test_results.append(test_result)
        test_dict[d] = test_results
    return test_dict


def test_combination_matrices(normalized_combined_distributions: List[NormalizedMatrix],
                              centroids: List[Centroid],
                              zones: List[Zone],
                              testing_data: List[Sample],
                              combination_method: Callable) -> Dict[NormalizedMatrix, TestResult]:
    combine_vectors = combination_method

    test_results = dict()  # type: Dict[NormalizedMatrix, TestResult]
    for normalized in normalized_combined_distributions:

        resultant = normalized.parent_matrix

        test_result = TestResult()

        # JC-01 used the measured/reported zone instead of the actual zone to the algorithm.
        for sample in testing_data:
            vectors = list()  # type: List[Dict[Zone, float]]
            answers = list()
            for distribution in resultant.normalizations:
                ap_rssi_dict = sample.get_ap_rssi_dict(*distribution.access_points)

                coord = get_NNv4_RSSI(centroid_points=centroids, rssis=ap_rssi_dict)
                # coord = get_NNv4(centroid_points=centroids, rssis=ap_rssi_dict)
                zone = get_zone(zones=zones, co_ordinate=coord)

                vector = distribution.get_vector(zone)
                vectors.append(vector)
                answers.append(zone)

            NormalizedMatrix.theAnswer = sample.answer  # JC-01 - used to pass the true answer around for run-time validation - used by dbg_combine_vector

            resultant_vector = combine_vectors(answers, *vectors)
            test_result.record(sample.answer, resultant_vector)

        test_results[normalized] = test_result

    return test_results


def test_svm_matrices(
        normalized_distribution: NormalizedMatrix,
        svm: svm_model,
        zones: List[Zone],
        test_class: List[int],
        test_features: List[Dict[AccessPoint, int]]) -> TestResult:
    test_result = TestResult()
    ap_test_features = list()
    for features in test_features:
        filtered_features = [value for key, value in features.items() if key in normalized_distribution.access_points]
        ap_test_features.append(filtered_features)
    p_labs, p_acc, p_vals = svm_predict(y=test_class, x=ap_test_features, m=svm, options="-q")
    for index, prediction in enumerate(p_labs):
        predict_zone = zones[int(prediction) - 1]
        actual_zone = zones[test_class[index] - 1]
        vector = normalized_distribution.get_vector(predict_zone)
        test_result.record(zone=actual_zone, vector=vector)
    return test_result
