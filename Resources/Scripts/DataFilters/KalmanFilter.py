from src.Resources.Scripts.DataFilters.SharedFilterResources import __ScanValue
from typing import List, Dict


def kalman_filter_data(raw_data_file_path: str,
                       input_variance: float = 50.0,
                       noise: float = 0.008) -> Dict[str, __ScanValue]:

    # Start from the RMAC-Filtered folders.


    # measurement_noise = # variance of RSSIs


    pass


# public class Kalman {
#
# /* Complete calculation of Kalman Filter */
# public static Double kalman (ArrayList<Double> inputValues, double initialVariance, double noise){
#     return calculate(inputValues, initialVariance, noise);
# }
#
# /* Calculation of Kalman Filter using default values for wireless Access Points data acquisition */
# public static Double kalman (ArrayList<Double> inputValues){
#     return calculate(inputValues, 50.0, 0.008);
# }
#
# /* Calculation of arithmetic mean */
# public static Double mean (ArrayList<Double> inputValues){
#     return StatUtils.mean(inputValues);
# }
#
#
# /*This method is the responsible for calculating the value refined with Kalman Filter */
# private static Double calculate(ArrayList<Double> inputValues, double initialVariance, double noise){
#     Double kalmanGain;
#     Double variance = initialVariance;
#     Double processNoise = noise;
#     Double measurementNoise = StatUtils.variance(inputValues);
#     Double mean = inputValues.get(0);
#
#     for (Double value : inputValues){
#         variance = variance + processNoise;
#         kalmanGain = variance/((variance+measurementNoise));
#         mean = mean + kalmanGain*(value - mean);
#         variance = variance - (kalmanGain*variance);
#     }
#
#     return mean;
# }
#
# public class StatUtils {
#
# static Double variance (ArrayList<Double> values){
#     Double sum = 0.0;
#     Double mean = mean(values);
#     for(double num : values){
#         sum += Math.pow(num - mean , 2);
#     }
#     return sum/(values.size()-1);
# }
#
# static Double mean (ArrayList<Double> values){
#     return sum(values)/values.size();
# }
#
# private static Double sum (ArrayList<Double> values){
#     Double sum = 0.0;
#     for (Double num : values){
#         sum+=num;
#     }
#     return sum;
# }
# }