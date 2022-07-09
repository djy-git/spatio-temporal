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
import metrics
from ydj.util import *
import argparse

def evaluate(settings):
    # type: (dict) -> float
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
    Returns:
        A score
    """
    predictions = []
    grounds = []
    raw_data_lst = []

    common_cols = ['TurbID', 'Day', 'Tmstamp']
    prediction_df = pd.read_csv(join(PATH.output, settings['filename']))
    ground_df = pd.read_csv(PATH.target)[common_cols + ['Patv']].rename(columns={'Patv': 'Patv_target'})
    raw_data_df = pd.read_csv(PATH.target)

    # assert len(sys.argv) == 2, "RIGHT USAGE: 'python evaluate.py {submission_file_name}'"
    for i in range(1,settings["capacity"]+1):
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
    day_len = settings["day_len"]
    day_acc = []
    for idx in range(0, preds.shape[0]):
        acc = 1 - metrics.rmse(preds[idx, -day_len:, -1], gts[idx, -day_len:, -1]) / (settings["capacity"] * 1000)
        if acc != acc:
            continue
        day_acc.append(acc)
    day_acc = np.array(day_acc).mean()
    print('Accuracy:  {:.4f}%'.format(day_acc * 100))
    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    # out_len = settings["output_len"]
    # mae, rmse = metrics.regressor_scores(predictions[:, -out_len:, :] / 1000, grounds[:, -out_len:, :] / 1000)

    overall_mae, overall_rmse = metrics.regressor_detailed_scores(predictions, grounds, raw_data_lst, settings)

    print('\n \t RMSE: {}, MAE: {}'.format(overall_rmse, overall_mae))

    # if settings["is_debug"]:
    #     end_test_time = time.time()
    #     print("\nElapsed time for evaluation is {} secs\n".format(end_test_time - end_forecast_time))

    total_score = (overall_mae + overall_rmse) / 2
    return total_score


if __name__ == "__main__":
    # Set up the initial environment
    # Current settings for the model
    parser = argparse.ArgumentParser(description='Long Term Wind Power Forecasting')
    ###
    parser.add_argument('--filename', type=str, default='baseline1.csv',
                        help='Filename of the input data, change it if necessary')
    parser.add_argument('--output_len', type=int, default=288, help='The length of predicted sequence')
    parser.add_argument('--day_len', type=int, default=144, help='Number of observations in one day')
    parser.add_argument('--capacity', type=int, default=134, help="The capacity of a wind farm, "
                                                                  "i.e. the number of wind turbines in a wind farm")
    parser.add_argument('--stride', type=int, default=1, help='The stride that a window adopts to roll the test set')
    args = parser.parse_args()
    settings = {
        # "data_path": args.data_path,
        "filename": args.filename,
        # "task": args.task,
        # "target": args.target,
        # "checkpoints": args.checkpoints,
        # "input_len": args.input_len,
        "output_len": args.output_len,
        # "start_col": args.start_col,
        # "in_var": args.in_var,
        # "out_var": args.out_var,
        "day_len": args.day_len,
        # "train_size": args.train_size,
        # "val_size": args.val_size,
        # "test_size": args.test_size,
        # "total_size": args.total_size,
        # "lstm_layer": args.lstm_layer,
        # "dropout": args.dropout,
        # "num_workers": args.num_workers,
        # "train_epochs": args.train_epochs,
        # "batch_size": args.batch_size,
        # "patience": args.patience,
        # "lr": args.lr,
        # "lr_adjust": args.lr_adjust,
        "capacity": 134,
        # "turbine_id": args.turbine_id,
        # "pred_file": args.pred_file,
        "stride": 1
        # "is_debug": args.is_debug
    }
    print('\n File Name : \n\t{}\n'.format(args.filename))

    score = evaluate(settings)
    print('\n --- Overall Score --- \n\t{}'.format(score))
