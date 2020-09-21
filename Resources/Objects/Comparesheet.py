from Resources.Objects.Building import Building
from Resources.Objects.Floor import Floor
from Resources.Objects.Matrices.NormalizedDistribution import sort_matrices, NormalizedMatrix
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.TestData import TestResult
from xlsxwriter import worksheet, workbook
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import pandas as pd


class Comparesheet:
    def __init__(self,
                 num_combinations: int,
                 error_mode: str,
                 k_fold: int,
                 building: Building,
                 floor: Floor,
                 best_gamma: str,
                 train_data: int,
                 test_data: int,
                 type_mode: str,
                 access_points: List[int],
                 tables: Dict[
                     int, List[Dict[
                         Tuple[str, ...], Tuple[
                             List[Dict[Tuple[AccessPoint], Tuple[float, float]]], List[float], float]]]]):

        self.__title = ""  # type: str
        self.__tab_title = ""  # type: str
        self.__best_gamma = best_gamma
        self.__num_combinations = num_combinations
        self.__error_mode = error_mode
        self.__tables = tables
        self.__k_fold = k_fold
        self.__train_data = train_data
        self.__test_data = test_data
        self.__building = building
        self.__floor = floor
        self.__access_points = access_points
        self.__type_mode = type_mode
        self.__column_width = len(self.__floor.zones)  # type: int
        self.__headers = ["Comparing Table"]
        self.__data_type = ["AP", "Beacon", "Mix"]
        self.__combination_modes = [('SVM', 'SVM'), ('SVM', 'NNv4'), ('SVM', 'kNNv1'), ('SVM', 'kNNv2'),
                                    ('SVM', 'kNNv3'), ('NNv4', 'SVM'), ('NNv4', 'NNv4'), ('NNv4', 'kNNv1'),
                                    ('NNv4', 'kNNv2'), ('NNv4', 'kNNv3'), ('kNNv1', 'SVM'), ('kNNv1', 'NNv4'),
                                    ('kNNv1', 'kNNv1'), ('kNNv1', 'kNNv2'), ('kNNv1', 'kNNv3'), ('kNNv2', 'SVM'),
                                    ('kNNv2', 'NNv4'), ('kNNv2', 'kNNv1'), ('kNNv2', 'kNNv2'), ('kNNv2', 'kNNv3'),
                                    ('kNNv3', 'SVM'), ('kNNv3', 'NNv4'), ('kNNv3', 'kNNv1'), ('kNNv3', 'kNNv2'),
                                    ('kNNv3', 'kNNv3')]
        self.__specific_modes = [('SVM', 'SVM'), ('SVM', 'NNv4'), ('NNv4', 'NNv4'), ('NNv4', 'SVM')]

    @property
    def title(self) -> str:
        if self.__title == "":
            self.__title = "ALL - {} train data - {} test data - GD Approach vs Joseph Method - {} Error Mode - {} Combination Mode - {} tables".format(
                self.__train_data,
                self.__test_data,
                self.__error_mode,
                len(self.__specific_modes),
                len(self.__tables.values()))

        return self.__title

    @property
    def tab_title(self) -> str:
        if self.__tab_title == "":
            self.__tab_title = "{} - {} - ALL".format(
                self.__building.building_name,
                self.__floor.floor_id)

            if len(self.__tab_title) > 31:
                self.__tab_title = self.__tab_title[:30]

        return self.__tab_title

    @staticmethod
    def save_key_page(sheet: worksheet, data_results: Dict[str, Dict[AccessPoint, Tuple[int, int, int]]], **formats) -> None:
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
        sheet.write("D10", "{ Building }")
        sheet.write("D11", "{ Floor }")
        sheet.write("D12", "{ Data type }")
        sheet.write("D13", "{ Sheet Type }")

        # -- Middle column:
        sheet.write("F9", "Example:", bold)
        sheet.write("F10", "Home, SCAET")
        sheet.write("F11", "1 5 6")
        sheet.write("F12", "AP, Mix, ALL")
        sheet.write("F13", "Matrix, Chart, Error, Table")

        # -- Right column:
        sheet.write("H9", "Meaning:", bold)
        sheet.write("H10", "Which building the data belong to")
        sheet.write("H11", "Which floor the data belong to")
        sheet.write("H12", "the signal type used")
        sheet.write("H13", "the information type ")

        for floor, data_result in data_results.items():
            if floor == "5":
                number = 16
            else:
                number = 26
            sheet.merge_range('D{}:H{}'.format(number - 1, number - 1), "Home - {}".format(floor), merge_format)
            sheet.write("D{}".format(str(number)), "Type", bold)
            sheet.write("E{}".format(str(number)), "AP", bold)
            sheet.write("F{}".format(str(number)), "Min", bold)
            sheet.write("G{}".format(str(number)), "Mean", bold)
            sheet.write("H{}".format(str(number)), "Max", bold)
            for ap, data in data_result.items():
                sheet.write("D{}".format(str(ap.num + number)), ap.type)
                sheet.write("E{}".format(str(ap.num + number)), ap.id)
                sheet.write("F{}".format(str(ap.num + number)), data[0])
                sheet.write("G{}".format(str(ap.num + number)), data[1])
                sheet.write("H{}".format(str(ap.num + number)), data[2])

    def save(self, sheet: worksheet, chart_sheet: worksheet, special_sheet: worksheet, book: workbook,
             **formats) -> None:
        # Formats:
        # bold = formats["bold"]
        bold = book.add_format({'bold': True})
        merge_format = formats["merge_format"]

        # Set spacing:
        horizontal_gap = 17
        vertical_gap = 11

        # # Write the header:
        sheet.merge_range('A1:X1', self.title, merge_format)
        chart_sheet.merge_range('A1:X1', self.title, merge_format)
        special_sheet.merge_range('A1:X1', self.title, merge_format)
        for index, table in self.__tables.items():
            num_ap = self.__access_points[index]
            # set charts
            avg_chart = book.add_chart({'type': 'line'})
            best_chart = book.add_chart({'type': 'line'})
            gd_col = index * horizontal_gap + 2
            jc_col = index * horizontal_gap + 5
            ig_col = index * horizontal_gap + 8
            mm_col = index * horizontal_gap + 11
            gd_tables = table[0]
            jc_tables = table[1]
            mm_tables = table[2]
            ig_tables = table[3]
            for modes, gd_tuple in gd_tables.items():
                table_chart = book.add_chart({'type': 'column'})
                mse_chart = book.add_chart({'type': 'line'})
                table_num = self.__specific_modes.index(modes)
                mode_name = modes[0] + "-" + modes[1]
                jc_tuple = jc_tables[modes]
                mm_tuple = mm_tables[modes]
                ig_tuple = ig_tables[modes]
                jc_results = jc_tuple[0]
                gd_results = gd_tuple[0]
                mm_results = mm_tuple[0]
                ig_results = ig_tuple[0]
                gd_train_time = gd_tuple[2]
                gd_test_time = gd_tuple[1]
                jc_train_time = jc_tuple[2]
                jc_test_time = jc_tuple[1]
                ig_train_time = ig_tuple[2]
                ig_test_time = ig_tuple[1]
                mm_train_time = mm_tuple[2]
                mm_test_time = mm_tuple[1]
                # first Row
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap, "{}".format(mode_name), bold)

                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 1, "GD Approach", bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 2, round(gd_train_time, 4), bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 3, round(sum(gd_test_time), 4), bold)

                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 4, "JC Method", bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 5, round(jc_train_time, 4), bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 6, round(sum(jc_test_time), 4), bold)

                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 7, "InfoGain Method", bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 8, round(ig_train_time, 4), bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 9, round(sum(ig_test_time), 4), bold)

                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 10, "MaxMean Method", bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 11, round(mm_train_time, 4), bold)
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap + 12, round(sum(mm_test_time), 4),
                            bold)

                # Second Row
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap, "Num of AP", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 1, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 2, "Accuracy", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 3, "Mean Error", bold)

                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 4, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 5, "Accuracy", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 6, "Mean Error", bold)

                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 7, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 8, "Accuracy", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 9, "Mean Error", bold)

                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 10, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 11, "Accuracy", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 12, "Mean Error", bold)

                # Rest Row
                length = num_ap
                mm_values = list()
                mm_keys = list()
                mm_mse = list()
                ig_keys = list()
                ig_values = list()
                ig_mse = list()
                gd_keys = list()
                gd_values = list()
                gd_mse = list()
                jc_keys = list()
                jc_values = list()
                jc_mse = list()
                # if modes[0] != "SVM" and modes[0] == modes[1] and index != 2:
                #     length += 3
                row_start = table_num * vertical_gap + 4
                gd_row_end = table_num * vertical_gap + 1 + num_ap
                jc_row_end = table_num * vertical_gap + 1 + length
                for d in range(3, length + 1):
                    sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap, "d={}".format(d), bold)
                    if d <= num_ap:
                        jc_result = jc_results[d - 3]
                        gd_result = gd_results[d - 3]
                        for key, value in gd_result.items():
                            gd_keys.append(key)
                            gd_values.append(value[0])
                            gd_mse.append(value[1])
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 1,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 2,
                                        round(value[0] * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 3,
                                        round(value[1], 4), bold)
                        for key, value in jc_result.items():
                            jc_keys.append(key)
                            jc_values.append(value[0])
                            jc_mse.append(value[1])
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 4,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 5,
                                        round(value[0] * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 6,
                                        round(value[1], 4), bold)
                        mm_result = mm_results[d - 3]
                        ig_result = ig_results[d - 3]
                        for key, value in ig_result.items():
                            ig_keys.append(key)
                            ig_values.append(value[0])
                            ig_mse.append(value[1])
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 7,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 8,
                                        round(value[0] * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 9,
                                        round(value[1], 4), bold)
                        for key, value in mm_result.items():
                            mm_keys.append(key)
                            mm_values.append(value[0])
                            mm_mse.append(value[1])
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 10,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 11,
                                        round(value[0] * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 12,
                                        round(value[1], 4), bold)
                    else:
                        jc_result = jc_results[d - 3]
                        for key, value in jc_result.items():
                            jc_keys.append(key)
                            jc_values.append(value[0])
                            jc_mse.append(value[1])
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 4,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 5,
                                        round(value[0] * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 2, index * horizontal_gap + 6,
                                        round(value[1], 4), bold)
                # End Row
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap, "Best", bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap, "Average", bold)

                best_gd_value = max(gd_values)
                best_gd_key = gd_keys[gd_values.index(best_gd_value)]
                best_gd_mse = min(gd_mse)
                average_gd_value = sum(gd_values) / len(gd_values)
                average_gd_mse = sum(gd_mse) / len(gd_mse)
                # average_gd_time = sum(gd_test_time) / len(gd_test_time)

                best_jc_value = max(jc_values)
                best_jc_mse = min(jc_mse)
                best_jc_key = jc_keys[jc_values.index(best_jc_value)]
                average_jc_value = sum(jc_values) / len(jc_values)
                average_jc_mse = sum(jc_mse) / len(jc_mse)
                # average_jc_time = sum(jc_test_time) / len(jc_test_time)

                best_ig_value = max(ig_values)
                best_ig_mse = min(ig_mse)
                best_ig_key = ig_keys[ig_values.index(best_ig_value)]
                average_ig_value = sum(ig_values) / len(ig_values)
                average_ig_mse = sum(ig_mse) / len(ig_mse)

                best_mm_value = max(mm_values)
                best_mm_mse = min(mm_mse)
                best_mm_key = mm_keys[mm_values.index(best_mm_value)]
                average_mm_value = sum(mm_values) / len(mm_values)
                average_mm_mse = sum(mm_mse) / len(mm_mse)

                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 1,
                            "{}".format(best_gd_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 2,
                            round(best_gd_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 3,
                            round(best_gd_mse, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 4,
                            "{}".format(best_jc_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 5,
                            round(best_jc_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 6,
                            round(best_jc_mse, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 7,
                            "{}".format(best_ig_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 8,
                            round(best_ig_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 9,
                            round(best_ig_mse, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 10,
                            "{}".format(best_mm_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 11,
                            round(best_mm_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 12,
                            round(best_mm_mse, 4), bold)

                # sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 1,
                #             "Average Values", bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 2,
                            round(average_gd_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 3,
                            round(average_gd_mse, 4), bold)
                # sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 5,
                #             "Average Values", bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 5,
                            round(average_jc_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 6,
                            round(average_jc_mse, 4), bold)
                # sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 9,
                #             "Average Values", bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 8,
                            round(average_mm_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 9,
                            round(average_mm_mse, 4), bold)
                # sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 12,
                #             "Average Values", bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 11,
                            round(average_ig_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length, index * horizontal_gap + 12,
                            round(average_ig_mse, 4), bold)

                table_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 4],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, jc_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, jc_col, jc_row_end, jc_col],
                })

                table_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 1],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, gd_col, gd_row_end, gd_col],
                })
                table_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 7],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, ig_col, gd_row_end, ig_col],
                })
                table_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 10],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, mm_col, gd_row_end, mm_col],
                })

                table_chart.set_x_axis({'name': 'D Value'})
                table_chart.set_y_axis({'name': 'Accuracy'})
                table_chart.set_y_axis({'max': 100, 'min': 50})
                table_chart.set_title({'name': '{} comparison with {}'.format(mode_name, self.__data_type[index])})
                chart_sheet.insert_chart(table_num * 20 + 5, index * 8, table_chart)

                mse_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 4],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, jc_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, jc_col + 1, jc_row_end, jc_col + 1],
                })

                mse_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 1],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, gd_col + 1, gd_row_end, gd_col + 1],
                })
                mse_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 7],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, ig_col + 1, gd_row_end, ig_col + 1],
                })
                mse_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 10],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, mm_col + 1, gd_row_end, mm_col + 1],
                })

                mse_chart.set_x_axis({'name': 'D Value'})
                mse_chart.set_y_axis(
                    {'name': 'Mean Error(m)'})
                mse_chart.set_title({'name': '{} comparison with {}'.format(mode_name, self.__data_type[index])})
                special_sheet.insert_chart(table_num * 20 + 5, index * 8, mse_chart)
