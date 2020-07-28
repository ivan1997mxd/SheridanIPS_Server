from Resources.Objects.Matrices.NormalizedDistribution import NormalizedMatrix
from Algorithms.Combination.Combination import get_combination_function
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint import GridPoint
from Resources.Objects.Points.Centroid import Centroid
from Offline.MatrixProduction import create_all_matrices_from_rssi_data
from Resources.Objects.TestData import create_test_data_list
from Resources.Objects.Zone import get_all_zones
from Resources.Objects.Points.Point import Point
from Resources.Objects.Worksheet import Worksheet
from Algorithms.Combination.AdaptiveBoosting import create_matrices
from typing import List
from time import time
from uuid import uuid4
import xlsxwriter.exceptions
import xlsxwriter


# TODO: Update Key page to include all acronyms and meanings.
# TODO: Also change the order pages are saved into workbook. Sort them before actually saving.


# Globals to set parameters for Matrix Production
dates = [
    #["November 19"], ["November 20"], ["November 21"], ["November 23"],
   ["November 19", "November 20", "November 21"],
   #["November 19"],
    #["November 19", "November 20", "November 21", "November 23"]
]
#times = ["19_00"]     # readonly
times = ["15_00", "17_00", "19_00"]     # readonly
num_combinations = [2, 3]
combination_modes = ["WGT"] # ["AVG", "WGT", "AB"]
error_modes = ["MAX"]
#error_modes = ["MAX", "DGN"]

# combination modes = "AVG", "WGT", "AB", for "Averaged", "Weighted", and "Adaptive Boosting", respectively.
# error modes = "DGN", "MAX", for "diagonal-based", and "max-based", respectively.


def automate_matrix_build(dates: List[List[str]],
                          times: List[str],
                          num_combinations: List[int],
                          combination_modes: List[str],
                          error_modes: List[str]):

    # raise Exception("The matrix files have already been produced. Only run this if a change has been made.")

    # No matter what the user selects, the Point objects never really change.
    # Instantiate them here, to avoid constantly opening and closing the CSV data files.

    # 1. Establish file location data.
    main_folder = "../Data/November"
    access_point_file_path = "{}/Points/Access Points/Access Points.csv".format(main_folder)
    grid_point_file_path = "{}/Points/Grid Points/November 19 - November 20 - November 21 - November 23 Grid Points.csv".format(main_folder)
    centroid_file_path = "{}/Points/Centroid Points/Centroid Points.csv".format(main_folder)
    zone_file_path = "{}/Zones/Zones.csv".format(main_folder)
    sorted_offline_rssi_folder_path = "{}/RSSI Data/Test Data/Offline/".format(main_folder)
    sorted_online_rssi_folder_path = "{}/RSSI Data/Test Data/Online/".format(main_folder)

    # 2. Instantiate "static" objects.
    access_points = AccessPoint.create_point_list(file_path=access_point_file_path)
    grid_points = GridPoint.create_point_list(file_path=grid_point_file_path, access_points=access_points)
    centroids = Centroid.create_point_list(file_path=centroid_file_path, grid_points=grid_points)
    zones = get_all_zones(file_path=zone_file_path)

    # 3. Start producing Matrices:
    worksheets = list()  # type: List[Worksheet]
    sheet_counter = 1
    num_worksheets = len(combination_modes) * len(error_modes) * len(dates) * len(num_combinations)
    print("There will be a total of {} worksheet{}.".format(num_worksheets, "" if num_worksheets == 1 else "s"))

    # 4. Start with dates to reduce the number of times test data needs to be read from the CSV files.
    for date_subset in dates:

        # 5. Get test data.
        training_data = create_test_data_list(access_points=access_points,
                                              zones=zones,
                                              folder_path=sorted_offline_rssi_folder_path,
                                              dates=date_subset,
                                              times=times)

        testing_data = create_test_data_list(access_points=access_points,
                                             zones=zones,
                                             folder_path=sorted_online_rssi_folder_path,
                                             dates=date_subset,
                                             times=times)

        print("-- Instantiated test lists.")
        print("-- Offline data size: {}".format(len(training_data)))
        print("-- Online data size: {}".format(len(testing_data)))

        # 6. Start combinations:
        for combination_mode in combination_modes:

            if combination_mode == "AB":
                combination_method = get_combination_function("WGT")

                for error_mode in error_modes:

                    NormalizedMatrix.error_mode = error_mode

                    for combination in num_combinations:
                        print("Working on sheet {} of {}.".format(sheet_counter, num_worksheets))
                        sheet_counter += 1
                        distributions = create_matrices(
                            access_points=access_points,
                            centroids=centroids,
                            zones=zones,
                            training_data=training_data,
                            testing_data=testing_data,
                            combination_method=combination_method,
                            num_combinations=combination)

                        # Separate the Tuple retrieved above.
                        probability_distributions = distributions[0]
                        normalized_distributions = distributions[1]
                        resultant_combinations = distributions[2]
                        normalized_resultant_combinations = distributions[3]
                        test_results = distributions[4]

                        worksheets.append(Worksheet(num_combinations=combination,
                                                    date_subset=date_subset,
                                                    error_mode=error_mode,
                                                    combination_mode="AB",
                                                    normalized_probability_matrices=normalized_distributions,
                                                    normalized_combined_matrices=normalized_resultant_combinations,
                                                    test_results=test_results))

                        # Reset the point objects:
                        Point.reset_points()

                continue

            # Set combination mode:
            combination_method = get_combination_function(combination_mode)

            for error_mode in error_modes:

                # Set error mode:
                NormalizedMatrix.error_mode = error_mode

                for combination in num_combinations:

                    print("Working on sheet {} of {}.".format(sheet_counter, num_worksheets))
                    sheet_counter += 1

                    distributions = create_all_matrices_from_rssi_data(
                        access_points=access_points,
                        centroids=centroids,
                        zones=zones,
                        training_data=training_data,
                        testing_data=testing_data,
                        combination_method=combination_method,
                        num_combinations=combination)

                    # Separate the Tuple retrieved above.
                    probability_distributions = distributions[0]
                    normalized_distributions = distributions[1]
                    resultant_combinations = distributions[2]
                    normalized_resultant_combinations = distributions[3]
                    test_results = distributions[4]

                    worksheets.append(Worksheet(num_combinations=combination,
                                                date_subset=date_subset,
                                                error_mode=error_mode,
                                                combination_mode=combination_mode,
                                                normalized_probability_matrices=normalized_distributions,
                                                normalized_combined_matrices=normalized_resultant_combinations,
                                                test_results=test_results))

                    # Reset the point objects:
                    Point.reset_points()

    # Start saving the workbook:
    excel_start_time = time()
    excel_workbook = xlsxwriter.Workbook("{}/Matrices/Results_old.xlsx".format(main_folder))
    bold = excel_workbook.add_format({'bold': True})
    merge_format = excel_workbook.add_format({'bold': True, 'align': 'center'})

    # Save the key page:
    excel_worksheet = excel_workbook.add_worksheet("Keys")
    Worksheet.save_key_page(excel_worksheet, bold=bold, merge_format=merge_format)

    # Save all other pages:
    problems_saving = list()    # type: List[str]
    for sheet in worksheets:
        try:
            excel_worksheet = excel_workbook.add_worksheet(sheet.tab_title)
        except xlsxwriter.exceptions.DuplicateWorksheetName:

            problem_sheet = sheet.tab_title
            problem_resolution = str(uuid4())[:25]
            problem_description = "Worksheet {} has the same tab-title as another page. It has been replaced with {}."
            problems_saving.append(problem_description.format(problem_sheet, problem_resolution))

            excel_worksheet = excel_workbook.add_worksheet(problem_resolution)

        sheet.save(excel_worksheet, bold=bold, merge_format=merge_format)
    excel_workbook.close()
    excel_end_time = time()

    print("Workbook Write Time: {}s.".format(excel_end_time - excel_start_time))

    if len(problems_saving) > 0:
        print("There were problems saving {} worksheets.".format(len(problems_saving)))
        for problem in problems_saving:
            print(problem)
        print("You can manually change the tab-titles now if desired.")


start_time = time()
automate_matrix_build(dates=dates,
                      times=times,
                      num_combinations=num_combinations,
                      combination_modes=combination_modes,
                      error_modes=error_modes)
end_time = time()

print("Total run time: {}".format(end_time - start_time))
