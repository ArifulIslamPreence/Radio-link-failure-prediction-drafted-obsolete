import argparse
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LogisticRegression

from rocket import generate_kernels, apply_kernels

df = pd.read_csv('train_data2.csv')


def train_eval_split(df: pd.DataFrame, column: str, train_ratio: float = 0.7):
    if train_ratio > 1:
        raise ValueError(f'train ration cannot be {train_ratio}')

    train_df = pd.DataFrame([], columns=df.columns)
    test_df = pd.DataFrame([], columns=df.columns)
    for c in set(df[column]):
        temp = df.loc[df[column] == c]
        # temp = temp.sort_values(by='datetime').reset_index(drop=True)
        temp = temp.sample(temp.shape[0]).reset_index(drop=True)

        l = temp.shape[0]
        n = int(l * train_ratio)

        train_df = pd.concat([train_df, temp.iloc[:n]], axis=0)
        test_df = pd.concat([test_df, temp.iloc[n:]], axis=0)

    return train_df, test_df


train_df, test_df = train_eval_split(df=df, column='month')


# target = 'rlf'
#
# x = train_df.loc[:, train_df.columns != target].values
# y = train_df.loc[:, train_df.columns == target].values.ravel()
#
# x = np.array(x).astype(float)
# y = np.array(y).astype(float)
#
# x_test = test_df.loc[:, test_df.columns != target].values
# y_test = test_df.loc[:, test_df.columns == target].values.ravel()
#
# x_test = np.array(x_test).astype(float)
# y_test = np.array(y_test).astype(float)


def run(training_data, test_data, num_runs=100, num_kernels=1000):
    results = np.zeros(num_runs)

    Y_training, X_training = training_data[:, 0].astype(np.int), training_data[:, 1:]
    Y_test, X_test = test_data[:, 0].astype(np.int), test_data[:, 1:]

    for i in range(num_runs):
        input_length = X_training.shape[1]
        kernels = generate_kernels(input_length, num_kernels)

        X_training_transform = apply_kernels(X_training, kernels)

        X_test_transform = apply_kernels(X_test, kernels)

        classifier = LogisticRegression(random_state=0, normalize=True)
        classifier.fit(X_training_transform, Y_training)

        results[i] = classifier.score(X_test_transform, Y_test)

    return results


df3 = pd.read_csv("train_data2.csv")

_ = generate_kernels(100, 1000)
apply_kernels(np.zeros_like(train_df)[:, 1:], _)

results = run(train_df, test_df,
              num_runs=100,
              num_kernels=1000)

print(results)
