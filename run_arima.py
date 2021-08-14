import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import csv

# class ARIMA:
#     def __init__(self, path_to_node_csv):
#         ## IMPORT DATA
#         self.df = pd.read_csv(path_to_node_csv, index_col ='round', parse_dates = True)
#         # df.head()

#         ## TRAIN/TEST SPLIT
#         self.split = int(0.75 * len(self.df))
#         self.train = self.df.iloc[:self.split]
#         self.test = self.df.iloc[self.split:]
#         self.test_4c = self.test.values.reshape(len(self.test),)

#         # CHECK BEST ARIMA MODEL
#         from pmdarima import auto_arima
#         import warnings

#         warnings.filterwarnings("ignore") # Ignore harmless warnings
#         self.stepwise_fit = auto_arima(self.train, start_p = 0, start_q = 0,
#                                 max_p = 1, max_q = 1, m = 5,
#                                 start_P = 0, seasonal = False,
#                                 d = None, D = 1, trace = True,
#                                 error_action ='ignore',
#                                 suppress_warnings = True,
#                                 stepwise = True)
#         # self.stepwise_fit.summary()

#         # pattern = rf'(\d)'
#         # match_object = re.search(pattern, str(self.stepwise_fit), flags=re.DOTALL)
#         # matches = re.findall(pattern, str(self.stepwise_fit))
#         # order = (int(matches[3]), int(matches[4]), int(matches[5]))
#         # print('---------', order)


#         self.n_periods = len(self.test)
#         self.fc, self.confint = self.stepwise_fit.predict(n_periods = self.n_periods, return_conf_int=True)

#         # ## BUILD ARIMA MODEL
#         # from statsmodels.tsa.arima_model import ARIMA  
#         # order = (0, 1, 0)
#         # model = ARIMA(self.train, 
#         # order=order)
#         # self.result = model.fit(disp=0)
#         # # result.summary()

#         # ## PREDICTION ARIMA MODEL AGAINST THE TEST SET
#         # self.result.plot_predict(dynamic=False)
#         # plt.show()

#     def future_forecast(self, show=False, save=False, title="Forecast for t rounds"):
#         index_of_fc = np.arange(len(self.train), len(self.train) + self.n_periods)

#         # make series for plotting purpose
#         fc_series = pd.Series(self.fc, index=index_of_fc)
#         lower_series = pd.Series(self.confint[:, 0], index=index_of_fc)
#         upper_series = pd.Series(self.confint[:, 1], index=index_of_fc)

#         # Plot
#         plt.figure(figsize=(15.5, 11.5))
#         axes = plt.gca()
#         axes.set_ylim([-2, 2])
#         plt.plot(self.df.value)
#         plt.plot(fc_series, color='darkgreen')
#         plt.fill_between(lower_series.index, 
#                         lower_series, 
#                         upper_series, 
#                         color='k', alpha=.15)
#         plt.title(title, fontdict={
#             'fontsize': 15,
#             # 'horizontalalignment': 'left'
#         })
#         if save:
#             plt.savefig(save['path'])
#         if not show:
#             plt.close()

#     def get_forecast(self):
#         return self.fc

#     from statsmodels.tsa.stattools import acf

#     # Accuracy Metrics
#     def forecast_accuracy(self):
#         # fc, se, conf = self.result.fc(len(self.test), alpha=0.05)  # 95% conf
#         mape = np.mean(np.abs(self.fc - self.test_4c)/np.abs(self.test_4c))  # MAPE
#         me = np.mean(self.fc - self.test_4c)             # ME
#         mae = np.mean(np.abs(self.fc - self.test_4c))    # MAE
#         mpe = np.mean((self.fc - self.test_4c)/self.test_4c)   # MPE
#         rmse = np.mean((self.fc - self.test_4c)**2)**.5  # RMSE
#         corr = np.corrcoef(self.fc, self.test_4c)[0,1]   # corr
#         mins = np.amin(np.hstack([self.fc[:,None], 
#                                 self.test_4c[:,None]]), axis=1)
#         maxs = np.amax(np.hstack([self.fc[:,None], 
#                                 self.test_4c[:,None]]), axis=1)
#         minmax = 1 - np.mean(mins/maxs)             # minmax
#     #     acf1 = acf(fc-test)[1]                      # ACF1
#         return f'MAPE: {mape}\nME: {me}\nMAE: {mae}\nMPE: {mpe}\nRMSE: {rmse}' #\nCORR: {corr}\nMINMAX: {minmax}'
#     #     return ({'mape':mape, 'me':me, 'mae': mae, 
#     #             'mpe': mpe, 'rmse':rmse,
#     # #             'acf1':acf1, 
#     #             'corr':corr, 'minmax':minmax})

def stepwise_fit(train_data):
     # CHECK BEST ARIMA MODEL
    from pmdarima import auto_arima
    import warnings
    warnings.filterwarnings("ignore") # Ignore harmless warnings
    return auto_arima(train_data, start_p = 0, start_q = 0,
                            max_p = 1, max_q = 1, m = 5,
                            start_P = 0, seasonal = False,
                            d = None, D = 1, trace = True,
                            error_action ='ignore',
                            suppress_warnings = True,
                            stepwise = True)

def make_segmented_predictions(this_actual_csv_path, this_prediction_csv_path, step, node_ids=['0', '1', '2', '4', '6', '8']):
    import copy
    import math
    from collections import OrderedDict
    import numpy as np
    import re

    df = pd.read_csv(this_actual_csv_path, index_col='node')
    battoir_df = copy.deepcopy(df)
    if len(df.columns) < 2 * step:
        raise Exception('columns length must be at least twice the step')

    # How many divisions along columns (segments) in this table
    segments_length = math.ceil(len(battoir_df.columns)/step)
    segments = OrderedDict()

    # Open CSV and make writer
    this_arima_predict_csv = open(this_prediction_csv_path.replace('.csv', f'_step-{step}.csv'), mode = 'w')
    this_arima_predict_csv_writer = csv.writer(this_arima_predict_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Make segments divisions and add to segments OrderedDict
    for i in range(1, segments_length + 1):
        if (len(battoir_df.columns) - (i-1) * step) >= step:
            segments[f'segment_{i}'] = battoir_df.iloc[:, (i-1) * step : i * step]
        elif (len(battoir_df.columns) - (i-1) * step) == 0:
            break
        else:
            segments[f'segment_{i}'] = battoir_df.iloc[:, (i-1) * step :]
    
    # Predicted CSV header with indexes
    iheader = []
    # Predicted CSV rows to write
    rows = OrderedDict()
    # Initialize all rows with an empty list
    for i in range(1, len(df) + 1):
        rows[f'row_{i}'] = []

    # For each segment, 
    for k, v in segments.items():
        # Predict each row and add as (1/segments_length) to rows OrderedDict
        i = 1
        for _, crow in segments[k].iterrows():
            fc, confint = stepwise_fit(crow).predict(n_periods = step, return_conf_int=True)
            rows[f'row_{i}'].extend(fc.tolist())
            i = i + 1

        # Find (1/segments_length) iheader
        cheader = []
        for index, _ in segments[k].T.iterrows():
            pattern = r'\d+'
            match = re.search(pattern, index)
            cheader.append(int(match.group(0)))

        # # Find the max of all the rows to know where the next set of columns will begin
        # header_start = int(np.max(cheader)) + 1
        # cheader = [f'round_{i}' for i in range(header_start, header_start + step)]

        # Extend iheader List with cheader as (1/segments_length)
        iheader.extend(cheader)

    # Make the header
    header_start = int(np.min(iheader)) + step
    header_stop = int(np.max(iheader)) + step
    header = ['node'] + [f'round_{i}' for i in range(header_start, header_stop + 1)]



    # Write the header
    this_arima_predict_csv_writer.writerow(header)


    # Write all the rows
    i = 0 # node indes start
    for k, v in rows.items():
        this_arima_predict_csv_writer.writerow([node_ids[i]] + rows[k])
        i = i + 1

    # Close the predicted csv file
    this_arima_predict_csv.close()
    

# def make_prediction_results(this_mlc_path):
#     this_arima_result_path = os.path.join(this_mlc_path, 'arima')
#     if not os.path.exists(this_arima_result_path):
#         os.makedirs(this_arima_result_path)
#     for f in os.listdir(this_mlc_path):
#         if f.endswith('.csv'):
#             this_csv_path = os.path.join(this_mlc_path, f)
#             result = ARIMA(this_csv_path)
#             this_csv_png_result_path = os.path.join(this_arima_result_path, f.replace('.csv', '.png'))
#             result.future_forecast(save={'path': this_csv_png_result_path}, title=result.forecast_accuracy())
    

# make_prediction_results(r'results\2021-08-07\09-03\remaining_energies\MLC')
# make_prediction_results(r'results\2021-08-07\20-49\remaining_energies\MLC')

# result = ARIMA(r'C:\Users\sanis\Desktop\sdwsn-new\results\2021-08-07\02-33\remaining_energies\MLC\node-99_remaining_energies.csv')
# result.future_forecast()
# print(result.forecast_accuracy())

make_segmented_predictions(r'C:\Users\sanis\Desktop\sdwsn-new\results\2021-08-13\15-05\remaining_energies\MLC\arima\aggregate\actual_remaining_energies.csv', r'C:\Users\sanis\Desktop\sdwsn-new\results\2021-08-13\15-05\remaining_energies\MLC\arima\aggregate\predicted_remaining_energies.csv', 3)