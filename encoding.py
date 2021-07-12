import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import TimeSeriesSplit
import csv
import copy
from datetime import date, timedelta

df1 = pd.read_csv("input_dataset/rl-kpis.csv", index_col=0, low_memory=False)
df2 = pd.read_csv("input_dataset/rl-sites.csv")
df3 = pd.read_csv("output_dataset/new_combined_forecast_v4.csv")

rl_kpis_history = df1.merge(df2[['site_id', 'groundheight', 'clutter_class']], on='site_id')

##mean_encoded = rl_kpis_history.groupby(['clutter_class'])['groundheight'].mean().to_dict()
# rl_kpis_history['clutter_class'] = rl_kpis_history['clutter_class'].map(mean_encoded)

df3['forecast_datetime'] = [pd.Timestamp(x) for x in df3['forecast_datetime']]
rl_kpis_history.columns = ['history_{}'.format(column) for column in rl_kpis_history.columns]
rl_kpis_history['history_datetime'] = [pd.Timestamp(x) for x in rl_kpis_history['history_datetime']]
rl_kpis_history.sort_values(by='history_datetime')
merged_df = df3.merge(rl_kpis_history, left_on=['mlid', 'forecast_datetime'],
                      right_on=['history_mlid', 'history_datetime'])
merged_df.to_csv('final2.csv')
# merged_df['polarization'] = merged_df['polarization'].fillna('None')
# merged_df['history_polarization'] = merged_df['history_polarization'].fillna('None')
# merged_df['freq_band'] = merged_df['freq_band'].fillna('None')
# merged_df['history_freq_band'] = merged_df['history_freq_band'].fillna('None')
# merged_df['history_clutter_class'] = merged_df['history_clutter_class'].fillna('None')
# merged_df['scalibility_score'] = merged_df['scalibility_score'].fillna(-1)
# merged_df['history_scalibility_score'] = merged_df['history_scalibility_score'].fillna(-1)
# df = merged_df.copy()  # df = merged_df.dropna()
# print(df.head())
#
# columns_to_ohe = ['card_type', 'freq_band', 'modulation',
#                   'history_card_type', 'history_freq_band', 'history_modulation',
#                   'history_clutter_class']
# df = copy.deepcopy(df)
# for c in columns_to_ohe:
#     temp = pd.get_dummies(df[c], prefix=c).astype('int')
#     df = df.drop(columns=c)
#     df = pd.concat([df, temp], axis=1)
#
# # df.to_csv("train_sample.csv")
# dfx = copy.deepcopy(df)
# we1 = {'heavy thunderstorm with rain showers': 1,
#        'heavy rain showers': 2,
#        'clear sky': 3,
#        'few clouds': 4,
#        'foggy': 5,
#        'heavy rain': 6,
#        'heavy snow': 7,
#        'hot day': 8,
#        'light intensity shower rain': 9,
#        'light rain': 10,
#        'light rain showers': 11,
#        'misty': 12,
#        'overcast cloud': 13,
#        'rain': 14,
#        'scattered clouds': 15,
#        'sleet': 16,
#        'snow': 17,
#        'thunderstorm with heavy rain': 18,
#        'windy': 19}
#
# dfx['weather_day1'] = dfx['weather_day1'].map(we1)
# dfx.drop(columns=['neid', 'direction', 'history_neid', 'history_direction'])
# dfx.to_csv('output_dataset/train_data.csv')
#
# df = pd.read_csv("output_dataset/train_data.csv")
# df['month'] = [pd.Timestamp(x).month for x in df['datetime']]
#
# df['humidity_max_day1'] = df['humidity_max_day1'].fillna('-1')
# df['humidity_min_day1'] = df['humidity_min_day1'].fillna('-1')
# df['wind_dir_day1'] = df['wind_dir_day1'].fillna('-1')
# df['wind_speed_day1'] = df['wind_speed_day1'].fillna('-1')
#
# # dropping unimportant features
# to_drop = ['datetime', 'mlid', 'mw_connection_no',
#            'site_id', 'scalibility_score', 'severaly_error_second', 'error_second', 'avail_time',
#            'forecast_datetime', 'station_no',
#            #  'temp_max_day1', 'temp_min_day1', 'humidity_max_day1', 'humidity_min_day1',
#            #  'wind_dir_day1', 'wind_speed_day1', 'wd1_clear sky', 'wd1_few clouds',
#            #  'wd1_foggy', 'wd1_heavy rain showers','wd1_heavy thunderstorm with rain showers', 'wd1_hot day',
#            #  'wd1_light intensity shower rain', 'wd1_light rain showers',
#            #  'wd1_misty', 'wd1_overcast clouds', 'wd1_scattered clouds', 'wd1_sleet',
#            #  'wd1_thunderstorm with heavy rain'
#            'datetime_ws', 'history_mlid', 'history_mw_connection_no',
#            'history_site_id', 'history_scalibility_score', 'history_avail_time',
#            'history_datetime'
#            ]
#
# df.drop(columns=to_drop)
#
# data = pd.read_csv('output_dataset/train_data.csv')
# data_types = pd.DataFrame(
#     data.dtypes,
#     columns=['Data Type']
# )
#
# missing_data = pd.DataFrame(
#     data.isnull().sum(),
#     columns=['Missing Values']
# )
# unique_values = pd.DataFrame(
#     columns=['Unique Values']
# )
# for row in list(data.columns.values):
#     unique_values.loc[row] = [data[row].nunique()]
#
# dq_report = data_types.join(missing_data).join(unique_values)
# dq_report.to_csv("dq_report.csv")
#
# data = pd.read_csv('train_data4.csv')
# tscv = TimeSeriesSplit(n_splits=12)
# features = data.columns
# X = pd.DataFrame(data, columns=features)
# y = data.rlf.values
# for train_index, test_index in tscv.split(data):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# from sklearn import preprocessing
#
# df = pd.read_csv('output_dataset/new_combined_forecast_2.csv')
# df_copy = df.copy()
# con = ['type', 'tip', 'direction', 'polarization', 'card_type', 'adaptive_modulation', 'freq_band', 'modulation',
#        'weather_day1']
# df_copy_con = df_copy[
#     ['type', 'tip', 'direction', 'polarization', 'card_type', 'adaptive_modulation', 'freq_band', 'modulation',
#      'weather_day1']]
# # df_norm = preprocessing.scale(df_copy_con)
# # df_norm_con = pd.DataFrame(df_norm)
# # df_norm_con.columns = con
#
# df_copy_con.hist(bins=30)
#
# # apply normalization techniques by Column 1
#
#
# # view normalized data
#
#
# # import numpy as np
# # import math
# #
# # X = DF['datetime']
# #
# # Y1 = DF['bbe']
# # Y2 = DF['rlf']
# # Y3 = DF['unavail_second']
# # Y4 = DF['link_length']
# #
# # figure, axis = plt.subplots(4, 1)
# #
# # axis[0, 0].plot(X, Y1)
# # axis[0, 0].set_title("bbe")
# #
# # axis[0, 1].plot(X, Y2)
# # axis[0, 1].set_title("rlf")
# #
# # axis[1, 0].plot(X, Y3)
# # axis[1, 0].set_title("unavailable second")
# #
# # axis[1, 1].plot(X, Y4)
# # axis[1, 1].set_title("link_length")
# #
# # # Combine all the operations and display
# # plt.show()
