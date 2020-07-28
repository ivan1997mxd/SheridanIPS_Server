from typing import List, Dict, Tuple, Union
from Objects.Collector import *
from Objects.Scan import Scan
from Matrices.Matrix import Matrix


class ProbabilityDistribution(Matrix):

    def __init__(self, trained_model: Union[IndividualModel, Tuple[AccessPoint, ...]]):

        if isinstance(trained_model, tuple):
            super(ProbabilityDistribution, self).__init__(access_points=trained_model)
            return

        self.__model = trained_model
        super(ProbabilityDistribution, self).__init__(access_points=trained_model.access_points)

    @property
    def model(self) -> IndividualModel:
        return self.__model


def build_probability_distribution_from_svm(ap_tuple: Tuple[AccessPoint, ...], svm: SVC, generation_data: List[Scan]):

    p_dist = ProbabilityDistribution(ap_tuple)

    # 1. Get all the predictions:
    training_features = list()  # type: List[List[int]]
    training_classes = list()  # type: List[List[int]]

    for scan in generation_data:

        rssi_features = list()

        for ap in ap_tuple:
            rssi_features.append(scan.get_rssi(ap))

        training_features.append(rssi_features)
        training_classes.append([scan.zone])

    predictions = svm.predict(training_features)

    for zone in [1, 2, 3, 4, 5]:
        zone_appearance_tracker = dict()    # type: Dict[int, int]
        samples_taken = 0

        for index, test_class in enumerate(generation_data):
            if test_class.zone == zone:
                prediction = predictions[index]

                if prediction in zone_appearance_tracker:
                    zone_appearance_tracker[prediction] += 1
                else:
                    zone_appearance_tracker[prediction] = 1

                # Update the samples tracker
                samples_taken += 1

            # Calculate the column of the Probability Matrix:
            for measured_zone, count in zone_appearance_tracker.items():
                # P(E | H) = (Number of times appeared + (1 / sample size)) / (sample size + 1)
                probability = (count * samples_taken + 1) / (samples_taken * (samples_taken + 1))
                p_dist.set_value(
                    measured_zone=measured_zone,
                    actual_zone=zone,
                    value=probability)

    return p_dist


# region External Constructor
def build_probability_distribution(trained_model: IndividualModel) -> ProbabilityDistribution:
    return build_probability_distributions([trained_model])[0]


def build_probability_distributions(trained_models: List[IndividualModel]) -> List[ProbabilityDistribution]:

    probability_matrices = list()   # type: List[ProbabilityDistribution]

    for model in trained_models:

        p_dist = ProbabilityDistribution(model)

        for zone in [1, 2, 3, 4, 5]:

            zone_appearance_tracker = dict()    # type: Dict[int, int]
            predicted_versus_actual = list()    # type: List[Tuple[int, int]]
            samples_taken = 0
            for index, test_class in enumerate(model.test_classes):
                if test_class == zone:
                    prediction = model.predictions[index]

                    if prediction in zone_appearance_tracker:
                        zone_appearance_tracker[prediction] += 1
                    else:
                        zone_appearance_tracker[prediction] = 1

                    # Update the samples tracker
                    samples_taken += 1

            # Calculate the column of the Probability Matrix:
            for measured_zone, count in zone_appearance_tracker.items():
                # P(E | H) = (Number of times appeared + (1 / sample size)) / (sample size + 1)
                probability = (count * samples_taken + 1) / (samples_taken * (samples_taken + 1))
                p_dist.set_value(
                    measured_zone=measured_zone,
                    actual_zone=zone,
                    value=probability)

        probability_matrices.append(p_dist)

    return probability_matrices


def __build_probability_distributions(grouped_tuples: List[CollectedModel]):

    probability_matrices = list()   # type: List[ProbabilityDistribution]

    for group in grouped_tuples:
        # Get best individual from the k-fold
        best_result = group.best_indy
        clf = best_result.clf

        probability_matrix = ProbabilityDistribution(access_points=[*best_result.access_point_tuple])

        for zone in [1, 2, 3, 4, 5]:

            # The model is already trained.
            zone_appearance_tracker = dict()    # type: Dict[int, int]
            predicted_versus_actual = list()    # type: List[Tuple[int, int]]
            samples_taken = 0
            for index, test_class in enumerate(best_result.test_classes):
                if test_class == zone:
                    prediction = best_result.predictions[index]

                    if prediction in zone_appearance_tracker:
                        zone_appearance_tracker[prediction] += 1
                    else:
                        zone_appearance_tracker[prediction] = 1

                    # Update the samples tracker
                    samples_taken += 1

            # Calculate the column of the Probability Matrix:
            for measured_zone, count in zone_appearance_tracker.items():
                # P(E | H) = (Number of times appeared + (1 / sample size)) / (sample size + 1)
                probability = (count * samples_taken + 1) / (samples_taken * (samples_taken + 1))
                probability_matrix.set_value(
                    measured_zone=measured_zone,
                    actual_zone=zone,
                    value=probability)

        probability_matrices.append(probability_matrix)

    return probability_matrices
# endregion
