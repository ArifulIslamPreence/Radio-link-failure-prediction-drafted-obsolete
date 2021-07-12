'''
combination of rl-kpis(radio) and met-real(weather station) dataset according to the minimum
distance between them.
'''

import pandas as pd
import numpy as np
import csv

f = open('input_dataset/siteVws.csv', 'r')
x_dict = dict()
for line in f:
    line = line.strip('\n').split(',')
    keys = line[1:]
    x_dict[keys[0]] = keys[1]

p = open('input_dataset/rl-kpis_mod.csv', 'r')
q = open('output_dataset/new_combined_forecast_2.csv', 'a')  # Generates Large dataset (30 gb + ) so its moved
# manually to the outside folder "Large Dataset"

header = 'Column1,type,datetime,tip,mlid,mw_connection_no,site_id,neid,direction,polarization,card_type,' \
         'adaptive_modulation,freq_band,link_length,severaly_error_second,error_second,unavail_second,avail_time,bbe,' \
         'rxlevmax,scalibility_score,capacity,modulation,rlf,forecast_datetime, Column1,station_no,datetime_ws,' \
         'weather_day1,' \
         'temp_max_day1,temp_min_day1,humidity_max_day1,humidity_min_day1,wind_dir_day1,wind_speed_day1'

q.write(header)
first_line = p.readline()

for line in p:
    line = line.strip('\n').split(',')
    keys = line[1:]
    site_id = keys[5]
    dateTime_r = keys[23]
    station_id = x_dict[site_id]
    a = open(r'input_dataset/met-forecast_day1.csv', 'r')
    lineStr = ','.join([str(elem) for elem in line])
    first_line = a.readline()

    for line1 in a:
        line1 = line1.strip('\n').split(',')
        item = line1[1]
        dateTime_s = line1[2]
        # print (item, station_id)

        if station_id == item and dateTime_s == dateTime_r:
            line1Str = ','.join([str(elem) for elem in line1])  # station_id being added separately in different line
            newline = lineStr + ',' + line1Str + '\n'
            q.write(newline)
