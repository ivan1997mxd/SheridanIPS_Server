from sklearn.svm import SVC

from Objects.FinalCombinationContainer import FinalCombinationContainer
from Objects.Scan import Scan
from Objects.TestData import TestResult
from Matrices.NormalizedDistribution import NormalizedDistribtuion
from Matrices.Matrix import Matrix
from Objects.AccessPoint import AccessPoint
from typing import List, Dict, Callable, Union, Tuple
import numpy as np


class CombinedMatrix(Matrix):
    # region Constructor
    def __init__(self, *normalizations: NormalizedDistribtuion):
        self.__normalizations = [*normalizations]      # type: List[NormalizedDistribtuion]
        self.__access_points = list()                  # type: List[AccessPoint]
        self.__zones = list()                          # type: List[int]
        self.__id = None                               # type: Union[None, str]

        for norm in self.__normalizations:
            for ap in norm.access_points:
                self.__access_points.append(ap)
            for zone in [1, 2, 3, 4, 5]:
                if zone not in self.__zones:
                    self.__zones.append(zone)

        super(CombinedMatrix, self).__init__(access_points=self.__access_points)
    # endregion

    # region Properties
    @property
    def normalizations(self) -> List[NormalizedDistribtuion]:
        return self.__normalizations
    # endregion

    # region Methods
    def increment_cell(self, actual_zone: int, resultant_vector: Dict[int, float]) -> None:
        # Find the highest probability, and check if it occurs more than once:
        highest_probability = max(resultant_vector.values())
        num_occurrences = 0

        for prob in resultant_vector.values():
            if prob == highest_probability:
                num_occurrences += 1

        # Find the zones being reported by the highest probabilities, and increment them:
        for zone, value in resultant_vector.items():
            if value == highest_probability:
                self.increment_value(zone, actual_zone, 1/num_occurrences)
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
                try:
                    aps += str(ap.num)
                except AttributeError:
                    aps += ap
            Str += aps + " U "
        self.__id = Str[:-3]
        return self.__id
    # endregion


def test_combination_matrices(
        final_combined_list: List[FinalCombinationContainer],
        test_features: List[Scan]) -> Dict[NormalizedDistribtuion, TestResult]:

    from Algorithms.Combination.Combination import weighted_combine_vectors

    combine_vectors = weighted_combine_vectors

    test_results = dict()           # type: Dict[NormalizedDistribtuion, TestResult]

    tests_complete = 1
    print("------ There are {} matrices to test.".format(len(final_combined_list)))
    print("------ Each matrix will be tested against {} samples.".format(len(test_features)))
    for final_combination in final_combined_list:
        print("------ Test #{}: {}.".format(tests_complete, final_combination.normalization.id))
        tests_complete += 1

        combined_dist = final_combination.normalization

        test_result = TestResult(final_combination)

        for test_index, scan in enumerate(test_features):

            vectors = list()
            answers = list()

            # We need to go through all the ap combinations
            ap_tuples = final_combination.ap_tuples

            # We need to get each SVM's prediction
            for ap_combination in ap_tuples:

                svm_list = final_combination.svm_list(ap_combination)

                predictions = list()    # type: List[List[float]]

                # For each SVM:
                for svm in svm_list:

                    aps_being_used = [x.num for x in ap_combination]
                    features = [x for index, x in enumerate(scan.rssis) if (index + 1) in aps_being_used]

                    predictions.append(svm.predict_proba([features]))

                # Average all the SVM's predictions:
                averaged_prediction = np.average(predictions, axis=0)[0]

                # Get the predicted zone (the index that matches the highest probability + 1).
                predicted_zone = np.where(averaged_prediction == np.amax(averaged_prediction))[0][0] + 1
                # Add 1 because the zone = index + 1.

                # Get the vector from the normalized distribution.
                vector = combined_dist.get_vector(predicted_zone)
                answers.append(predicted_zone)

                vectors.append(vector)

            resultant_vector = combine_vectors(answers, *vectors)
            test_result.record(scan.zone, resultant_vector)

        test_results[combined_dist] = test_result

    return test_results


def test_combination_matrices_no_k_fold(
    combined_dist_ap_dict: Dict[Tuple[AccessPoint, ...], CombinedMatrix],
    n_combined_dist_ap_dict: Dict[Tuple[AccessPoint, ...], NormalizedDistribtuion],
    svm_ap_holder: Dict[Tuple[AccessPoint, ...], SVC],
    test_features: List[Scan]) -> Dict[NormalizedDistribtuion, TestResult]:

    from Algorithms.Combination.Combination import weighted_combine_vectors

    combine_vectors = weighted_combine_vectors

    access_point_combos = list(combined_dist_ap_dict.keys())

    test_results = dict()                   # type: Dict[NormalizedDistribtuion, TestResult]
    for ap_combo in access_point_combos:

        n_c_dist = n_combined_dist_ap_dict[ap_combo]

        test_result = TestResult()

        for test_index, scan in enumerate(test_features):

            vectors = list()
            answers = list()

            for ap in ap_combo:

                aps_being_used = [x.num for x in ap]
                features = [x for index, x in enumerate(scan.rssis) if (index + 1) in aps_being_used]

                prediction = svm_ap_holder[ap].predict([features])[0]

                vector = n_c_dist.get_vector(prediction)

                vectors.append(vector)
                answers.append(prediction)

            resultant_vector = combine_vectors(answers, *vectors)
            test_result.record(scan.zone, resultant_vector)

        test_results[n_c_dist] = test_result

    return test_results

