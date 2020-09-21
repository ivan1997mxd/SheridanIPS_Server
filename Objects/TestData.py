from typing import Dict, Union

from Objects.FinalCombinationContainer import FinalCombinationContainer


class TestResult:

    def __init__(self, final_combination: Union[None, FinalCombinationContainer] = None):
        self.__correct = 0              # type: int
        self.__total_tests = 0          # type: int
        self.__sec_correct = 0              # type: int
        self.__answer_details = dict()  # type: Dict[int, Dict[str, int]]
        self.__final_combination = final_combination

    @property
    def final_combination(self):
        return self.__final_combination

    @property
    def answer_details(self):
        return self.__answer_details

    @property
    def accuracy(self) -> float:
        if self.__total_tests == 0:
            return 0
        return self.__correct / self.__total_tests

    @property
    def num_correct(self) -> int:
        return self.__correct

    @property
    def tests_ran(self) -> int:
        return self.__total_tests

    #JC-01 - add code to support 2nd guess information
    def record(self, zone: int, vector: Dict[int, float]) -> None:
        self.__total_tests += 1

        # Get max probability in the vector
        most_likely_zone = max(vector, key=vector.get)
        if most_likely_zone == zone:
            self.__correct += 1

            if zone not in self.__answer_details:
                self.__answer_details[zone] = {"times_tested": 1, "times_correct": 1, "times_2nd_correct" : 0}
            else:
                self.__answer_details[zone]["times_tested"] += 1
                self.__answer_details[zone]["times_correct"] += 1
            return

        vector[most_likely_zone] = 0.0
        most_likely_zone = max(vector, key=vector.get)
        if most_likely_zone == zone:
            self.__sec_correct += 1

            if zone not in self.__answer_details:
                self.__answer_details[zone] = {"times_tested": 1, "times_correct": 0, "times_2nd_correct" : 1}
            else:
                self.__answer_details[zone]["times_tested"] += 1
                self.__answer_details[zone]["times_2nd_correct"] += 1
            return

        if zone not in self.__answer_details:
            self.__answer_details[zone] = {"times_tested": 1, "times_correct": 0, "times_2nd_correct" : 0}
        else:
            self.__answer_details[zone]["times_tested"] += 1