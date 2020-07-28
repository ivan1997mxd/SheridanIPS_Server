from xlsxwriter import worksheet
from typing import List, Dict, Tuple
from Matrices.ProbabilityDistribution import ProbabilityDistribution
from Matrices.NormalizedDistribution import NormalizedDistribtuion, sort_matrices
from Objects.AccessPoint import AccessPoint
from Objects.TestData import TestResult


class Worksheet:

    # def __init__(self,
    #              fold_number: int,
    #              num_combinations: int,
    #              normalized_probability_matrices: List[NormalizedDistribtuion],
    #              normalized_combined_matrices: List[NormalizedDistribtuion],
    #              test_results: List[TestResult]):

    def __init__(self, num_combinations: int,
                 percent_reserved: float,
                 test_results: Dict[NormalizedDistribtuion, TestResult],
                 averaged_matrices: List[NormalizedDistribtuion] = None,
                 fold_number: int = -1,
                 use_k_fold: bool = True):

        self.__title = ""       # type: str
        self.__tab_title = ""   # type: str
        self.__fold_number = fold_number
        self.__test_results = test_results
        self.__percent_reserved = percent_reserved
        self.__num_combinations = num_combinations
        self.__column_width = 5
        self.__use_k_fold = use_k_fold

        if use_k_fold:
            # Averaged matrices after K-folding:
            self.__normalized_averaged_matrices = averaged_matrices

            # Combined matrices:
            self.__normalized_combined_matrices = test_results.keys()

            sort_matrices(matrix_list=list(self.__normalized_combined_matrices))
            sort_matrices(matrix_list=self.__normalized_averaged_matrices)

            self.__headers = ["Normalized Averaged Distributions", "Normalized Combined Distributions", "Test Results"]

        else:

            # Need to get distributions.
            self.__probability_distributions = list()
            self.__normalized_distributions = list()
            self.__combined_distributions = list()
            self.__n_c_distributions = list()

            for n_dist, test_result in test_results.items():
                self.__n_c_distributions.append(n_dist)

                combined_parent = n_dist.parent_matrix

                self.__combined_distributions.append(combined_parent)

                for norm in combined_parent.normalizations:

                    if norm in self.__normalized_distributions:

                        continue

                    self.__normalized_distributions.append(norm)

                    p_dist = norm.parent_matrix

                    self.__probability_distributions.append(p_dist)

            # Need to get test results.

    @property
    def title(self) -> str:
        if self.__title == "":
            self.__title = "{} Folds - {} Combinations - {} Reserved".format(self.__fold_number,
                                                                             self.__num_combinations,
                                                                             self.__percent_reserved)

        return self.__title

    @property
    def tab_title(self) -> str:
        if self.__tab_title == "":
            self.__tab_title = "{} FOLDS - {} COMBINATIONS".format(self.__fold_number, self.__num_combinations)

            if len(self.__tab_title) > 31:
                self.__tab_title = self.__tab_title[:30]

        return self.__tab_title

    @staticmethod
    def save_key_page(sheet: worksheet, **formats) -> None:
        # Formats:
        bold = formats["bold"]
        merge_format = formats["merge_format"]

        title = "KEYS"
        example_title = "{ Combinations } - { Dates } - { Error Mode } - { Combination Mode }"

        # Write the header:
        sheet.merge_range('D4:H4', title, merge_format)

        # Write the example page header:
        sheet.write("D6", "Examples:", bold)
        sheet.write("E6", example_title)

        # Write the key:
        # -- Left column:
        sheet.write("D9", "In title:", bold)
        sheet.write("D10", "{ Combinations }")
        sheet.write("D11", "{ Dates }")
        sheet.write("D12", "{ Error Mode }")
        sheet.write("D13", "{ Combination Mode }")

        # -- Middle column:
        sheet.write("F9", "Example:", bold)
        sheet.write("F10", "2 Combinations")
        sheet.write("F11", "N19, N20")
        sheet.write("F12", "WGT Error Mode")
        sheet.write("F13", "AB Combination Mode")

        # -- Right column:
        sheet.write("H9", "Meaning:", bold)
        sheet.write("H10", "Using a combination of 2 matrices.")
        sheet.write("H11", "Using data from November 19 (N19), and November 20 (N20).")
        sheet.write("H12", "Using weighted errors.")
        sheet.write("H13", "Using Adaptive Boosting combination method.")

    def save(self, sheet: worksheet, **formats) -> None:
        # Formats:
        bold = formats["bold"]
        merge_format = formats["merge_format"]

        # Set spacing:
        prob_horiz_spacing = self.__column_width + 2
        norm_horiz_spacing = prob_horiz_spacing + 1
        test_horizontal_spacing = prob_horiz_spacing + norm_horiz_spacing * 2 + 3
        matrix_vertical_spacing = self.__column_width + 5

        # Write the header:
        sheet.merge_range('A1:X1', self.title, merge_format)

        # Write the headers:
        if self.__use_k_fold:
            sheet.write(2, 0, "Probability Distributions", bold)
            sheet.write(2, norm_horiz_spacing, "Normalized Averaged Distributions", bold)
            sheet.write(2, norm_horiz_spacing * 2 + 1, "Normalized Combined Distributions", bold)
            sheet.write(2, test_horizontal_spacing, "Test Results", bold)
        else:
            sheet.write(2, 0, "Probability Distributions", bold)
            sheet.write(2, norm_horiz_spacing, "Normalized Distributions", bold)
            sheet.write(2, norm_horiz_spacing * 2 + 1, "Normalized Combined Distributions", bold)
            sheet.write(2, test_horizontal_spacing, "Test Results", bold)
            sort_matrices(matrix_list=self.__normalized_distributions)
            sort_matrices(matrix_list=self.__n_c_distributions)

            self.__normalized_averaged_matrices = self.__normalized_distributions
            self.__normalized_combined_matrices = self.__n_c_distributions

        # Write the probability distribution columns:
        for index, normalized_distribution in enumerate(self.__normalized_averaged_matrices):

            probability_distribution = normalized_distribution.parent_matrix

            # Probability distributions:
            for row, value_list in enumerate(probability_distribution.csv_list):
                for col, val in enumerate(value_list):
                    if row == 0:    # Matrix Header
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col, val, bold)
                    else:
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col, val)

            # Normalized probability distributions:
            for row, value_list in enumerate(normalized_distribution.csv_list):
                for col, val in enumerate(value_list):
                    if row == 0:    # Matrix Header
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col + norm_horiz_spacing, val, bold)
                    else:
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col + norm_horiz_spacing, val)

        # Write the normalized combined distribution column:
        norm_horiz_spacing *= 2
        for index, distribution in enumerate(self.__normalized_combined_matrices):
            for row, value_list in enumerate(distribution.csv_list):
                for col, val in enumerate(value_list):
                    if row == 0:    # Matrix Header
                        sheet.write(row + 3 + matrix_vertical_spacing * index,
                                    col + norm_horiz_spacing + 1, val, bold)
                    else:
                        sheet.write(row + 3 + matrix_vertical_spacing * index,
                                    col + norm_horiz_spacing + 1, val)

        #JC-01 add code to report 2nd guess accuracy
        # Write the test result column:
        for index, distribution in enumerate(self.__normalized_combined_matrices):
            result = self.__test_results[distribution]

            # Num tests ran:
            sheet.write(3 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        distribution.csv_list[0][0], bold)
            sheet.write(4 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Tests Ran:")
            sheet.write(4 + matrix_vertical_spacing * index,
                        test_horizontal_spacing + 1,
                        result.tests_ran)

            # Test Result Headers:
            sheet.write(5 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Zones:")
            sheet.write(6 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Times Tested:")
            sheet.write(7 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Times Correct:")
            sheet.write(8 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Zone Percentage Correct:")
            sheet.write(9 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Overall Percentage Correct:")
            sheet.write(9 + matrix_vertical_spacing * index,
                        test_horizontal_spacing + 1,
                        result.accuracy)
            sheet.write(10 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Times 2nd Correct:")
            sheet.write(11 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Zone Percentage with 2nd Correct:")

            # Test Result Data:
            with2 = 0
            percentage_eq = '=INDIRECT(ADDRESS(ROW()-1, COLUMN()))/INDIRECT(ADDRESS(ROW()-2,COLUMN()))'
            for zone, zone_results in result.answer_details.items():

                # Zone Numbers:
                sheet.write(5 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone, str(zone))

                # Times Tested:
                sheet.write(6 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone, zone_results["times_tested"])

                # Times Correct:
                sheet.write(7 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone, zone_results["times_correct"])

                # Zone correct percentage:
                sheet.write(8 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone,
                            zone_results["times_correct"] / zone_results["times_tested"])

                # Zone 2nd correct:
                sheet.write(10 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone, zone_results["times_2nd_correct"])

                # Zone correct percentage:
                sheet.write(11 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone,
                            (zone_results["times_correct"] + zone_results["times_2nd_correct"]) / zone_results["times_tested"])

                # No. of success including the 2nd times
                with2 = with2 + zone_results["times_correct"] + zone_results["times_2nd_correct"]

            sheet.write(12 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Overall Percentage with 2nd Correct:")
            sheet.write(12 + matrix_vertical_spacing * index,
                        test_horizontal_spacing + 1,
                        with2/result.tests_ran)
