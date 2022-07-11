# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Evaluate the performance
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
# import os
# import sys
# import time
# import traceback
# import numpy as np
from common import *
import metrics


def evaluate(prediction_df):
    # type: (dict) -> float
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
    Returns:
        A score
    """

    output_len = 288
    day_len= 144
    capacity =134
    stride = 1
    predictions = []
    grounds = []
    raw_data_lst = []

    common_cols = ['TurbID', 'Day', 'Tmstamp']

    ground_df = pd.read_csv(PATH.target)[common_cols + ['Patv']].rename(columns={'Patv': 'Patv_target'})
    raw_data_df = pd.read_csv(PATH.target)

    # assert len(sys.argv) == 2, "RIGHT USAGE: 'python evaluate.py {submission_file_name}'"
    for i in range(1,capacity+1):
        prediction = prediction_df[prediction_df['TurbID']==i].iloc[:,-1].to_numpy().reshape(1,-1,1)
        ground     = ground_df[ground_df['TurbID']==i].iloc[:,-1].to_numpy().reshape(1,-1,1)
        raw_data     = raw_data_df[raw_data_df['TurbID']==i]
        predictions.append(prediction)
        grounds.append(ground)
        raw_data_lst.append(raw_data)

    # start_forecast_time = time.time()
    #
    # predictions, grounds, raw_data_lst = forecast_module.forecast(settings)
    # end_forecast_time = time.time()
    # if settings["is_debug"]:
    #     print("\nElapsed time for prediction is: {} secs\n".format(end_forecast_time - start_forecast_time))

    preds = np.asarray(predictions)
    gts = np.asarray(grounds)
    preds = np.sum(preds, axis=0)
    gts = np.sum(gts, axis=0)

    # A convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    day_acc = []
    for idx in range(0, preds.shape[0]):
        acc = 1 - metrics.rmse(preds[idx, -day_len:, -1], gts[idx, -day_len:, -1]) / (capacity * 1000)
        if acc != acc:
            continue
        day_acc.append(acc)
    day_acc = np.array(day_acc).mean()
    print('Accuracy:  {:.4f}%'.format(day_acc * 100))
    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    # out_len = settings["output_len"]
    # mae, rmse = metrics.regressor_scores(predictions[:, -out_len:, :] / 1000, grounds[:, -out_len:, :] / 1000)

    overall_mae, overall_rmse = metrics.regressor_detailed_scores(predictions, grounds, raw_data_lst)

    print('\n \t RMSE: {}, MAE: {}'.format(overall_rmse, overall_mae))

    # if settings["is_debug"]:
    #     end_test_time = time.time()
    #     print("\nElapsed time for evaluation is {} secs\n".format(end_test_time - end_forecast_time))

    total_score = (overall_mae + overall_rmse) / 2
    return total_score


if __name__ == "__main__":

    filename = 'baseline1.csv'
    prediction_df = pd.read_csv(join(PATH.output, filename))
    print('\n File Name : \n\t{}\n'.format(filename))

    score = evaluate(prediction_df)
    print('\n --- Overall Score --- \n\t{}'.format(score))