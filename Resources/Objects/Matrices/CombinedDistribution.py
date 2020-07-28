from Algorithms.svm.svmutil import svm_predict
from Objects.FinalCombinationContainer import FinalCombinationContainer
from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix
from Resources.Objects.Matrices.Matrix import Matrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.TestData import Sample, TestResult
from Resources.Objects.Points.Centroid import Centroid
from Algorithms.NearestNeighbour.NNv4 import get_NNv4, get_NNv4_RSSI
from Resources.Objects.Zone import Zone, get_zone
from typing import List, Dict, Callable


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
        final_combined_list: List[FinalCombinationContainer],
        test_features: List[Sample],
        combination_method: Callable) -> Dict[NormalizedMatrix, TestResult]:
    combine_vectors = combination_method

    test_results = dict()  # type: Dict[NormalizedMatrix, TestResult]

    tests_complete = 1
    print("------ There are {} matrices to test.".format(len(final_combined_list)))
    print("------ Each matrix will be tested against {} samples.".format(len(test_features)))
    for final_combination in final_combined_list:
        print("------ Test #{}: {}.".format(tests_complete, final_combination.normalization.id))
        tests_complete += 1

        combined_dist = final_combination.normalization

        test_result = TestResult()

        for test_index, scan in enumerate(test_features):

            vectors = list()
            answers = list()

            # We need to go through all the ap combinations
            ap_tuples = final_combination.ap_tuples

            # We need to get each SVM's prediction
            for ap_combination in ap_tuples:

                svm_list = final_combination.svm_list(ap_combination)
                zones = final_combination.normalization.zones

                predictions = list()  # type: List[List[float]]

                # For each SVM:
                for svm in svm_list:
                    aps_being_used = [x for x in ap_combination]
                    features = [value for key, value in scan.scan.items() if key in aps_being_used]
                    # features = [x for key, x in scan.scan if (key + 1) in aps_being_used]
                    p_labs, p_acc, p_vals = svm_predict(y=[scan.answer.num], x=[features], m=svm, options="-q")
                    predictions.append(p_labs)

                best_predict_zone = max(predictions, key=predictions.count)
                predicted_zone = zones[int(best_predict_zone[0]) - 1]
                # Average all the SVM's predictions:
                # averaged_prediction = np.average(predictions, axis=0)[0]

                # Get the predicted zone (the index that matches the highest probability + 1).
                # predicted_zone = np.where(averaged_prediction == np.amax(averaged_prediction))[0][0] + 1
                # Add 1 because the zone = index + 1.

                # Get the vector from the normalized distribution.
                vector = combined_dist.get_vector(predicted_zone)
                answers.append(predicted_zone)

                vectors.append(vector)

            resultant_vector = combine_vectors(answers, *vectors)
            test_result.record(scan.answer, resultant_vector)

        test_results[combined_dist] = test_result

    return test_results
