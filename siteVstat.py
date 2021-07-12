#Minimum distanced weather station with rl

import csv
import numpy as np
import pandas as pd

df = pd.read_csv("distance_wrt_s.csv", index_col=0)
ws_name = []
site_name = []
min_val = []
min_val = df.idxmin()

for i,j in min_val.iteritems():
    site_name.append(i)
    ws_name.append(j)

tup = list(zip(site_name, ws_name))
df2 = pd.DataFrame(tup, columns=['site_id', 'station_id'])
df2.to_csv("siteVws.csv")
