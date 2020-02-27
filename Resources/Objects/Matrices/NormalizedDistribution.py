from Resources.Objects.Matrices.ProbabilityDistribution import ProbabilityMatrix
from Resources.Objects.Matrices.Matrix import Matrix
from Resources.Objects.Zone import Zone
from Resources.Objects.TestData import Sample
from typing import List, Iterable, Tuple, Union, Dict


class NormalizedMatrix(Matrix):

    error_mode = ""     # type: str
    theAnswer = ""      # JC-01 - used to hold the sample actual zone - sample.answer

    # region Constructor
    def __init__(self, matrix: Matrix, combine_ids: bool = False):
        super(NormalizedMatrix, self).__init__(access_points=matrix.access_points, zones=matrix.zones, size=matrix.size)

        self.__parent_matrix = matrix               # type: Matrix
        self.__csv_list = None                      # type: Union[None, List[List[str]]]
        self.__average_matrix_error = -1            # type: float
        self.__average_matrix_success = -1          # type: float
        self.__normalize_matrix(matrix)

        if combine_ids:
            Str = ""
            for child in self.__parent_matrix.normalizations:
                Str += child.id + " U "
            self.id = Str[:-3]
    # endregion

    # region Properties
    @property
    def average_matrix_error(self) -> float:
        if self.__average_matrix_error != -1:
            return self.__average_matrix_error

        error_sum = 0
        for measured_zone, vector in self.vectors:
            error_sum += self.__get_vector_error(measured_zone, vector)

        self.__average_matrix_error = error_sum / self.size
        return self.__average_matrix_error

    @property
    def average_matrix_success(self) -> float:
        return 1 - self.average_matrix_error

    # region Overrides:
    @property
    def csv_list(self) -> List[List[str]]:
        if self.__csv_list is not None:
            return self.__csv_list

        csv_list = list()   # type: List[List[str]]
        csv_list.append(["Access Point Combination: " + self.id])
        csv_list.append(["Zones"] + [str(x) for x in self.zones] + ["Error"] + ["Success"])

        for measured_zone, vector in self.vectors:
            csv_list.append([str(measured_zone)] +
                            [*vector.values()] +
                            [self.__get_vector_error(measured_zone, vector)] +
                            [self.get_vector_success(measured_zone, vector)])

        csv_list.append(["" for _ in range(self.size)] +
                        ["Averages:", self.average_matrix_error, self.average_matrix_success])
        self.__csv_list = csv_list

        return csv_list
    # endregion

    @property
    def parent_matrix(self) -> Matrix:
        return self.__parent_matrix
    # endregion

    # region Private Methods
    #JC - cross-relation
    @classmethod
    def __get_vector_error(cls, measured_zone: Zone, vector: Dict[Zone, float]) -> float:
        m = max(vector.values())
        a = vector[measured_zone]
        if NormalizedMatrix.error_mode == "DGN":
            return 1 - a #vector[measured_zone]
        elif NormalizedMatrix.error_mode == "MAX":
            return 1 - a #vector[measured_zone] #max(vector.values())

        raise NotImplementedError("Error mode {} is unknown".format(NormalizedMatrix.error_mode))

    @staticmethod
    def get_vector_success(measured_zone: Zone, vector: Dict[Zone, float]) -> float:
        # TODO: Re-implement this and get_vector_error properly.
        #return 1 - NormalizedMatrix.__get_vector_error(measured_zone, vector)
        m = max(vector.values())
        a = vector[measured_zone]
        if NormalizedMatrix.error_mode == "DGN":
            return a #vector[measured_zone]
        elif NormalizedMatrix.error_mode == "MAX":
            return m #vector[measured_zone] #max(vector.values())

        raise NotImplementedError("Error mode {} is unknown".format(NormalizedMatrix.error_mode))

    def __normalize_matrix(self, matrix: Matrix) -> None:
        for measured_zone, vector in matrix.measured_zones_and_vectors:

            row_sum = sum(vector.values())
            for actual_zone, value in vector.items():

                if row_sum == 0:
                    self.set_value(measured_zone=measured_zone, actual_zone=actual_zone, value=0)
                else:
                    self.set_value(measured_zone=measured_zone, actual_zone=actual_zone, value=value/row_sum)
    # endregion

    # region Comparison Operators
    def __lt__(self, other):
        return True if self.average_matrix_error < other.average_matrix_error else False

    def __le__(self, other):
        return True if self.average_matrix_error <= other.average_matrix_error else False

    def __gt__(self, other):
        return True if self.average_matrix_error > other.average_matrix_error else False

    def __ge__(self, other):
        return True if self.average_matrix_error >= other.average_matrix_error else False
    # endregion

    def test(self, sample: Sample) -> bool:
        pass


def build_normalized_distributions(probability_distributions: List[ProbabilityMatrix]) -> List[NormalizedMatrix]:
    return [NormalizedMatrix(p) for p in probability_distributions]


def sort_matrices(matrix_list: List[NormalizedMatrix] = None,
                  matrix_tuples: Iterable[Tuple[Matrix, Matrix]] = None
                  ) -> Union[List[Tuple[ProbabilityMatrix, NormalizedMatrix]], None]:
    if matrix_list is None and matrix_tuples is None:
        raise Exception("User must pass one list.")

    if matrix_list is not None and matrix_tuples is not None:
        raise Exception("User must use only one parameter.")

    if matrix_list is not None:
        # In place Selection sort
        for i in range(len(matrix_list)):
            min_index = i
            for j in range(i + 1, len(matrix_list)):
                if matrix_list[min_index] > matrix_list[j]:
                    min_index = j
            matrix_list[i], matrix_list[min_index] = matrix_list[min_index], matrix_list[i]
        return

    # Selection sort
    matrix_tuples = list(matrix_tuples)
    for i in range(len(matrix_tuples)):
        min_index = i
        for j in range(i + 1, len(matrix_tuples)):
            if matrix_tuples[min_index][1] > matrix_tuples[j][1]:
                min_index = j

        matrix_tuples[i], matrix_tuples[min_index] = matrix_tuples[min_index], matrix_tuples[i]
    return matrix_tuples
