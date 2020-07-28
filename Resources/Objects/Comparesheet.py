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
                         Tuple[str, ...], Tuple[List[Dict[Tuple[AccessPoint], float]], List[float], float]]]]):

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
        self.__specific_modes = [('SVM', 'SVM'), ('NNv4', 'NNv4'), ('kNNv1', 'kNNv1'), ('kNNv2', 'kNNv2'),
                                 ('kNNv3', 'kNNv3')]

    @property
    def title(self) -> str:
        if self.__title == "":
            self.__title = "ALL - {} train data - {} test data - GD Approach vs Joseph Method - {} Error Mode - {} Combination Mode - {} tables".format(
                self.__train_data,
                self.__test_data,
                self.__error_mode,
                len(self.__combination_modes),
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

    def save(self, sheet: worksheet, chart_sheet: worksheet, special_worksheet: worksheet, book: workbook,
             **formats) -> None:
        # Formats:
        bold = formats["bold"]
        merge_format = formats["merge_format"]

        # Set spacing:
        horizontal_gap = 13
        vertical_gap = 11

        # # Write the header:
        sheet.merge_range('A1:X1', self.title, merge_format)
        chart_sheet.merge_range('A1:X1', self.title, merge_format)
        special_worksheet.merge_range('A1:X1', self.title, merge_format)
        for index, table in self.__tables.items():
            jc_win = 0
            gd_win = 0
            rd_win = 0
            mm_win = 0
            gd_avg_win = 0
            jc_avg_win = 0
            rd_avg_win = 0
            mm_avg_win = 0
            draw = 0
            avg_draw = 0
            num_ap = self.__access_points[index]
            # set charts
            gd_chart = book.add_chart({'type': 'line'})
            jc_chart = book.add_chart({'type': 'line'})
            gd_col = index * horizontal_gap + 2
            jc_col = index * horizontal_gap + 5
            mm_col = index * horizontal_gap + 10
            rd_col = index * horizontal_gap + 8
            gd_tables = table[0]
            jc_tables = table[1]
            mm_tables = table[2]
            rd_tables = table[3]
            for modes, gd_tuple in gd_tables.items():
                table_chart = book.add_chart({'type': 'column'})
                table_num = self.__combination_modes.index(modes)
                mode_name = modes[0] + "-" + modes[1]
                jc_tuple = jc_tables[modes]
                mm_tuple = mm_tables[modes]
                rd_tuple = rd_tables[modes]
                jc_results = jc_tuple[0]
                gd_results = gd_tuple[0]
                mm_results = mm_tuple[0]
                rd_results = rd_tuple[0]
                gd_train_time = gd_tuple[2]
                gd_test_time = gd_tuple[1]
                jc_train_time = jc_tuple[2]
                jc_test_time = jc_tuple[1]
                mm_time = mm_tuple[1]
                rd_time = rd_tuple[1]
                # first Row
                sheet.write(table_num * vertical_gap + 2, index * horizontal_gap, "{}".format(mode_name), bold)
                sheet.merge_range(table_num * vertical_gap + 2, index * horizontal_gap + 1,
                                  table_num * vertical_gap + 2, index * horizontal_gap + 3,
                                  "GD Approach",
                                  merge_format)
                sheet.merge_range(table_num * vertical_gap + 2, index * horizontal_gap + 4,
                                  table_num * vertical_gap + 2, index * horizontal_gap + 6, "JC Method",
                                  merge_format)

                sheet.merge_range(table_num * vertical_gap + 2, index * horizontal_gap + 7,
                                  table_num * vertical_gap + 2, index * horizontal_gap + 8,
                                  "Random Method",
                                  merge_format)
                sheet.merge_range(table_num * vertical_gap + 2, index * horizontal_gap + 9,
                                  table_num * vertical_gap + 2, index * horizontal_gap + 10,
                                  "MaxMean Method",
                                  merge_format)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 7, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 8, "Accuracy", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 9, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 10, "Accuracy", bold)

                # Second Row
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap, "Num of AP", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 1, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 2, "Accuracy", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 3, "Time", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 4, "Ap Set", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 5, "Accuracy", bold)
                sheet.write(table_num * vertical_gap + 3, index * horizontal_gap + 6, "Time", bold)

                # Rest Row
                length = num_ap
                mm_values = list()
                mm_keys = list()
                rd_keys = list()
                rd_values = list()
                gd_keys = list()
                gd_values = list()
                jc_keys = list()
                jc_values = list()
                if modes[0] != "SVM" and modes[0] == modes[1] and index != 2:
                    length += 3
                row_start = table_num * vertical_gap + 4
                gd_row_end = table_num * vertical_gap + 2 + num_ap
                jc_row_end = table_num * vertical_gap + 2 + length
                for d in range(2, length + 1):
                    sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap, "d={}".format(d), bold)
                    if d <= num_ap:
                        jc_result = jc_results[d - 2]
                        gd_result = gd_results[d - 2]
                        for key, value in gd_result.items():
                            gd_keys.append(key)
                            gd_values.append(value)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 1,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 2,
                                        round(value * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 3,
                                        "{}s".format(round(gd_test_time[d - 2], 4)), bold)
                        for key, value in jc_result.items():
                            jc_keys.append(key)
                            jc_values.append(value)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 4,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 5,
                                        round(value * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 6,
                                        "{}s".format(round(jc_test_time[d - 2], 4)), bold)
                        mm_result = mm_results[d - 2]
                        rd_result = rd_results[d - 2]
                        for key, value in rd_result.items():
                            rd_keys.append(key)
                            rd_values.append(value)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 7,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 8,
                                        round(value * 100, 4), bold)
                        for key, value in mm_result.items():
                            mm_keys.append(key)
                            mm_values.append(value)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 9,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 10,
                                        round(value * 100, 4), bold)
                    else:
                        jc_result = jc_results[d - 2]
                        for key, value in jc_result.items():
                            jc_keys.append(key)
                            jc_values.append(value)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 4,
                                        "{}".format(key), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 5,
                                        round(value * 100, 4), bold)
                            sheet.write(table_num * vertical_gap + 3 + d - 1, index * horizontal_gap + 6,
                                        "{}s".format(round(jc_test_time[d - 2], 4)), bold)
                # End Row
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap, "Overall", bold)

                best_gd_avg = np.mean(gd_values)
                best_gd_value = max(gd_values)
                best_gd_key = gd_keys[gd_values.index(best_gd_value)]

                best_jc_avg = np.mean(jc_values)
                best_jc_value = max(jc_values)
                best_jc_key = jc_keys[jc_values.index(best_jc_value)]

                best_rd_avg = np.mean(rd_values)
                best_rd_value = max(rd_values)
                best_rd_key = rd_keys[rd_values.index(best_rd_value)]

                best_mm_avg = np.mean(mm_values)
                best_mm_value = max(mm_values)
                best_mm_key = mm_keys[mm_values.index(best_mm_value)]

                best_method = [best_gd_value, best_jc_value, best_rd_value, best_mm_value]
                best_avg_method = [best_gd_avg, best_jc_avg, best_rd_avg, best_mm_avg]
                winner = [best_method.index(w) for w in best_method if w == max(best_method)]
                avg_winner = [best_avg_method.index(a) for a in best_avg_method if a == max(best_avg_method)]
                if len(winner) == 1:
                    if winner[0] == 0:
                        gd_win += 1
                    elif winner[0] == 1:
                        jc_win += 1
                    elif winner[0] == 2:
                        rd_win += 1
                    else:
                        mm_win += 1
                else:
                    draw += 1
                if len(avg_winner) == 1:
                    if avg_winner[0] == 0:
                        gd_avg_win += 1
                    elif avg_winner[0] == 1:
                        jc_avg_win += 1
                    elif avg_winner[0] == 2:
                        rd_avg_win += 1
                    else:
                        mm_avg_win += 1
                else:
                    avg_draw += 1

                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 1,
                            "{}".format(best_gd_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 2,
                            round(best_gd_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 3,
                            "{}s".format(round(sum(gd_test_time), 4)), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 4,
                            "{}".format(best_jc_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 5,
                            round(best_jc_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 6,
                            "{}s".format(round(sum(jc_test_time), 4)), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 7,
                            "{}".format(best_rd_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 8,
                            round(best_rd_value * 100, 4), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 9,
                            "{}".format(best_mm_key), bold)
                sheet.write(table_num * vertical_gap + 2 + length + 1, index * horizontal_gap + 10,
                            round(best_mm_value * 100, 4), bold)

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
                    'values': [self.__tab_title, row_start, rd_col, gd_row_end, rd_col],
                })
                table_chart.add_series({
                    'name': [self.__tab_title, table_num * vertical_gap + 2, index * horizontal_gap + 9],
                    'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end,
                                   index * horizontal_gap],
                    'values': [self.__tab_title, row_start, mm_col, gd_row_end, mm_col],
                })

                table_chart.set_x_axis({'name': 'D Value'})
                table_chart.set_y_axis({'name': 'Accuracy'})
                table_chart.set_title({'name': '{} comparison with {}'.format(mode_name, self.__data_type[index])})
                chart_sheet.insert_chart(table_num * 20 + 10, index * 8, table_chart)

            chart_sheet.write(2, index * 8, "Comparison Table", bold)
            chart_sheet.write(2, index * 8 + 1, "GD Approach", bold)
            chart_sheet.write(2, index * 8 + 2, "JC Method", bold)
            chart_sheet.write(2, index * 8 + 3, "Random", bold)
            chart_sheet.write(2, index * 8 + 4, "MaxMean", bold)
            chart_sheet.write(2, index * 8 + 5, "Draw", bold)

            chart_sheet.write(3, index * 8, "Win single", bold)
            chart_sheet.write(3, index * 8 + 1, gd_win, bold)
            chart_sheet.write(3, index * 8 + 2, jc_win, bold)
            chart_sheet.write(3, index * 8 + 3, rd_win, bold)
            chart_sheet.write(3, index * 8 + 4, mm_win, bold)
            chart_sheet.write(3, index * 8 + 5, draw, bold)

            chart_sheet.write(4, index * 8, "Win avg", bold)
            chart_sheet.write(4, index * 8 + 1, gd_avg_win, bold)
            chart_sheet.write(4, index * 8 + 2, jc_avg_win, bold)
            chart_sheet.write(4, index * 8 + 3, rd_avg_win, bold)
            chart_sheet.write(4, index * 8 + 4, mm_avg_win, bold)
            chart_sheet.write(3, index * 8 + 5, avg_draw, bold)
            #     gd_chart.add_series({
            #         'name': [self.__tab_title, table_num * gap + 3, index * horizontal_gap],
            #         'categories': [self.__tab_title, row_start, index * horizontal_gap, gd_row_end, index * horizontal_gap],
            #         'values': [self.__tab_title, row_start, gd_col, gd_row_end, gd_col]
            #     })
            #
            #     jc_chart.add_series({
            #         'name': [self.__tab_title, table_num * gap + 3, index * horizontal_gap],
            #         'categories': [self.__tab_title, row_start, index * horizontal_gap, jc_row_end, index * horizontal_gap],
            #         'values': [self.__tab_title, row_start, jc_col, jc_row_end, jc_col]
            #     })
            # gd_chart.set_x_axis({'name': 'D Value'})
            # gd_chart.set_y_axis({'name': 'Accuracy'})
            # gd_chart.set_title({'name': 'Accuracy of 25 modes using GD Approach'})
            # jc_chart.set_x_axis({'name': 'D Value'})
            # jc_chart.set_y_axis({'name': 'Accuracy'})
            # jc_chart.set_title({'name': 'Accuracy of 25 modes using JC Method'})
            # # Insert the chart into the worksheet.
            # chart_sheet.insert_chart(len(self.__combination_modes) * 20 + 4 + num_ap, index * 8, gd_chart)
            # chart_sheet.insert_chart((len(self.__combination_modes) + 1) * 20 + 4 + num_ap, index * 8, jc_chart)
        # Write the headers:
        # sheet.write(2, 0, "Comparing Table", bold)
        # sheet.merge_range('B3:C3', self.__location_mode, bold)
        # sheet.merge_range('D3:E3', self.__mode, bold)
        # sheet.write(3, 0, "index", bold)
        # sheet.merge_range('B4:C4', "GD Approach", merge_format)
        # sheet.merge_range('D4:E4', "JC Method", merge_format)
        # sheet.write(4, 0, "d (2 < d < length of AP)", bold)
        # sheet.write(4, 1, "AP set result", bold)
        # sheet.write(4, 2, "Accuracy", bold)
        # sheet.write(4, 3, "AP set result", bold)
        # sheet.write(4, 4, "Accuracy", bold)
        # for key, value in self.__gd_result.items():
        #     length = len(key)
        #     sheet.write(length + 3, 0, "d={}".format(length))
        #     sheet.write(length + 3, 1, "{}".format(key))
        #     sheet.write(length + 3, 2, round(value, 4))
        # for key, value in self.__joseph_result.items():
        #     length = len(key)
        #     sheet.write(length + 3, 3, "{}".format(key))
        #     sheet.write(length + 3, 4, round(value, 4))
