from Resources.Objects.Matrices.NormalizedDistribution import sort_matrices, NormalizedMatrix
from Resources.Objects.Matrices.CombinedDistribution import CombinedMatrix
from Resources.Objects.Points.Centroid import Centroid
from Algorithms.NearestNeighbour.NNv4 import get_NNv4
from Resources.Objects.Zone import get_zone, Zone
from Resources.Objects.TestData import Sample
from typing import List, Dict, Tuple, Callable, Iterator
from itertools import combinations
import math
import pymongo

theAnswer = 1

def __nCr(n: int, r: int):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def get_combination_function(combination_mode: str) -> Callable:
    if combination_mode == "AVG":
        return __sum_combine_vectors
    elif combination_mode == "WGT":
        return __weighted_combine_vectors
    elif combination_mode == "DBG":
        return __dbg_combine_vectors
    else:
        raise Exception("The Combination Mode: {} is invalid.".format(combination_mode))


def __combinations(normalized_distributions: List[NormalizedMatrix],
                   num_combinations: int) -> Iterator[List[NormalizedMatrix]]:

    # Get combinations of matrices:
    primary_distribution = normalized_distributions[0]
    combination = combinations(normalized_distributions[1:], num_combinations - 1)

    num_combinations = __nCr(len(normalized_distributions) - 1, num_combinations - 1)

    print("-- There will be {} matrices produced.".format(num_combinations))

    for combo in combination:
        yield [primary_distribution] + [*combo]

#JC Used for training phase
def build_combined_distributions(centroids: List[Centroid],
                                 zones: List[Zone],
                                 normalized_distributions: List[NormalizedMatrix],
                                 training_data: List[Sample],
                                 combination_method: Callable,
                                 num_combinations: int,
                                 skip_good_classifiers: bool) -> Tuple[List[CombinedMatrix], List[NormalizedMatrix]]:
    client = pymongo.MongoClient(host='localhost', port=27017)
    db = client.SheridanTest
    OfflineData = db.OfflineData

    # Sort the matrices:
    sort_matrices(normalized_distributions)

    # Set combination method:
    combine_vectors = combination_method

    # Containers to hold results.
    combined_distributions = list()  # type: List[CombinedMatrix]
    normalized_combinations = list()  # type: List[NormalizedMatrix]

    # For every combination:
    for matrix_list in __combinations(normalized_distributions, num_combinations):

        # Empty matrix to hold the results of all tests.
        combined_matrix = CombinedMatrix(*matrix_list, size=matrix_list[0].size)

        for sample in training_data:

            vectors = list()        # type: List[Dict[Zone, float]]
            answers = list()
            for matrix in matrix_list:

                coord = get_NNv4(centroid_points=centroids,
                                 rssis=sample.get_ap_rssi_dict(*matrix.access_points))
                zone = get_zone(zones=zones, co_ordinate=coord)
                # print("find at " + str(zone) + " in matrix " + str(matrix.id))
                vector = matrix.get_vector(zone)
                vectors.append(vector)
                answers.append(zone)

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


def __sum_vectors(*vectors: Dict[Zone, float]) -> Dict[Zone, float]:
    dic = dict()        # type: Dict[Zone, float]
    for v in vectors:
        for actual_zone, value in v.items():
            if actual_zone not in dic.keys():
                dic[actual_zone] = value
            else:
                dic[actual_zone] += value
    return dic


def __sum_combine_vectors(answers: List[Zone], *vectors: Dict[Zone, float] ) -> Dict[Zone, float]:
    vector = __sum_vectors(*vectors)
    length = len(vectors)

    for zone, v in vector.items():
        vector[zone] = v / length

    return vector

def __weighted_combine_vectors(answers: List[Zone], *vectors: Dict[Zone, float]) -> Dict[Zone, float]:
    new_vectors = list()    # type: List[Dict[Zone, float]]
    for answer, vector in zip(answers, vectors):
        alpha = NormalizedMatrix.get_vector_success(answer, vector)
        new_vectors.append(NormalizedMatrix.scalar_vector(vector, alpha))

    return __sum_vectors(*new_vectors)

#JC-01 - add debugging code to dump out the involved vectors
def __dbg_combine_vectors(answers: List[Zone], *vectors: Dict[Zone, float]) -> Dict[Zone, float]:
    new_vectors = list()    # type: List[Dict[Zone, float]]
    for answer, vector in zip(answers, vectors):
        alpha = NormalizedMatrix.get_vector_success(answer, vector)
        new_vectors.append(NormalizedMatrix.scalar_vector(vector, alpha))
    v = __sum_vectors(*new_vectors)

    # if weightcombined work, then return
    z1 =  max(v, key=v.get)
    theAnswer= NormalizedMatrix.theAnswer
    if z1 == theAnswer:
        return v

    print("Debug....")
    z1s = " "
    for zone, i in v.items():
        z1s = z1s + str(i) + " ,"

    # not working, want to see what is the difference
    # focus on the best 2
    #make the 2nd best
    v[z1] = 0.0
    z2 = max(v, key=v.get)

    print("the Answer is " ,  theAnswer, " Weighted Answer is ", z1, " 2nd Zone is ", z2)
    print("Weighted Vector is ", z1s)

    #use the zone weight
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
