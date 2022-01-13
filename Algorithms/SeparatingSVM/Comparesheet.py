from Resources.Objects.Matrices.NormalizedDistribution import sort_matrices, NormalizedMatrix
from Resources.Objects.TestData import TestResult
from xlsxwriter import worksheet
from typing import List, Dict


class Worksheet:
    def __init__(self, num_combinations: int,
                 date_subset: List[str],
                 error_mode: str,
                 combination_mode: str,
                 location_mode: str,
                 normalized_probability_matrices: List[NormalizedMatrix],
                 normalized_combined_matrices: List[NormalizedMatrix],
                 test_results: Dict[NormalizedMatrix, TestResult]):

        self.__title = ""  # type: str
        self.__tab_title = ""  # type: str
        self.__num_combinations = num_combinations
        self.__date_subset = date_subset
        self.__error_mode = error_mode
        self.__combination_mode = combination_mode
        self.__location_mode = location_mode
        self.__normalized_probability_matrices = normalized_probability_matrices
        self.__normalized_combined_matrices = normalized_combined_matrices
        self.__test_results = test_results
        self.__column_width = self.__normalized_combined_matrices[0].size  # type: int

        self.__headers = ["Method 1", "Method 2", "Method 3"]

    @property
    def title(self) -> str:
        if self.__title == "":
            self.__title = "Compare result of three methods"

        return self.__title

    @property
    def tab_title(self) -> str:
        if self.__tab_title == "":
            self.__tab_title = "Method compare"

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
        sheet.write(2, 0, "Probability Distributions", bold)
        sheet.write(2, norm_horiz_spacing, "Normalized Probability Distributions", bold)
        sheet.write(2, norm_horiz_spacing * 2 + 1, "Normalized Combined Distributions", bold)
        sheet.write(2, test_horizontal_spacing, "Test Results", bold)


        # Write the probability distribution columns:
        for index, normalized_distribution in enumerate(self.__normalized_probability_matrices):

            probability_distribution = normalized_distribution.parent_matrix

            # Probability distributions:
            for row, value_list in enumerate(probability_distribution.csv_list):
                for col, val in enumerate(value_list):
                    if row == 0:  # Matrix Header
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col, val, bold)
                    else:
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col, val)

            # Normalized probability distributions:
            for row, value_list in enumerate(normalized_distribution.csv_list):
                for col, val in enumerate(value_list):
                    if row == 0:  # Matrix Header
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col + norm_horiz_spacing, val, bold)
                    else:
                        sheet.write(row + 3 + matrix_vertical_spacing * index, col + norm_horiz_spacing, val)

        # Write the normalized combined distribution column:
        norm_horiz_spacing *= 2
        for index, distribution in enumerate(self.__normalized_combined_matrices):
            for row, value_list in enumerate(distribution.csv_list):
                for col, val in enumerate(value_list):
                    if row == 0:  # Matrix Header
                        sheet.write(row + 3 + matrix_vertical_spacing * index,
                                    col + norm_horiz_spacing + 1, val, bold)
                    else:
                        sheet.write(row + 3 + matrix_vertical_spacing * index,
                                    col + norm_horiz_spacing + 1, val)

        # JC-01 add code to report 2nd guess accuracy
        # Write the test result column:
        for distribution, result in self.__test_results.items():
            index = list(self.__test_results.keys()).index(distribution)
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
                            test_horizontal_spacing + zone.num, str(zone))

                # Times Tested:
                sheet.write(6 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone.num, zone_results["times_tested"])

                # Times Correct:
                sheet.write(7 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone.num, zone_results["times_correct"])

                # Zone correct percentage:
                sheet.write(8 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone.num,
                            zone_results["times_correct"] / zone_results["times_tested"])

                # Zone 2nd correct:
                sheet.write(10 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone.num, zone_results["times_2nd_correct"])

                # Zone correct percentage:
                sheet.write(11 + matrix_vertical_spacing * index,
                            test_horizontal_spacing + zone.num,
                            (zone_results["times_correct"] + zone_results["times_2nd_correct"]) / zone_results[
                                "times_tested"])

                # No. of success including the 2nd times
                with2 = with2 + zone_results["times_correct"] + zone_results["times_2nd_correct"]

            sheet.write(12 + matrix_vertical_spacing * index,
                        test_horizontal_spacing,
                        "Overall Percentage with 2nd Correct:")
            sheet.write(12 + matrix_vertical_spacing * index,
                        test_horizontal_spacing + 1,
                        with2 / result.tests_ran)



