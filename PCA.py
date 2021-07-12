
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df1 = pd.read_csv("input_dataset/combined_short-main-2.csv", index_col=0, low_memory=False)
train, test = train_test_split(df1, test_size=0.30, random_state=0)
features = train.columns
batch_size = 100
dimension = len(features)

if np.any(np.isnan(train)):
    train.replace(np.nan, 0)

X = StandardScaler().fit_transform(train)

#PCA applied

pca_data = PCA(n_components=len(features) / 2)
post_pca_data = pca_data.fit_transform(X)
pca_Df = pd.DataFrame(data = post_pca_data
             , columns = ['feature {i}'.format(i) for i in range(len(features)/2)])

print(pca_Df.head(20))

import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df1 = pd.read_csv("combined_short-main-2.csv", index_col=0, low_memory=False)
train, test = train_test_split(df1, test_size=0.30, random_state=0)
features = train.columns
batch_size = 100
dimension = len(features)

if np.any(np.isnan(train)):
    train.replace(np.nan, 0)

X = StandardScaler().fit_transform(train)

#PCA applied

pca_data = PCA(n_components=len(features) / 2)
post_pca_data = pca_data.fit_transform(X)
pca_Df = pd.DataFrame(data = post_pca_data
             , columns = ['feature {i}'.format(i) for i in range(len(features)/2)])

print(pca_Df.head(20))
