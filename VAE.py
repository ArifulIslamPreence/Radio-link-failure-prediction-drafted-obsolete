'''Implementation of Variational Autoencoder Network for dataset reconstructing into normalized form.
 The whole combined dataset is fed into model by spliting batches'''

import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
import math
import matplotlib.pyplot as plt

df1 = pd.read_csv("output_dataset/new_combined.csv", index_col=0, low_memory=False)
train, test = train_test_split(df1, test_size=0.30, random_state=0)
features = train.columns
batch_size = 100
df1 = df1.interpolate(method='linear', limit_direction= 'forward')
train.fillna(train.mean(),inplace = True)
test.fillna(test.mean(),inplace = True)

#Train_data
normalizer = preprocessing.Normalizer(norm="l2")
training = normalizer.fit_transform(train)
training = pd.DataFrame(training, columns= features)
train_tensor = torch.tensor(training.values.astype(np.float32))
train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

#Test data
testing = normalizer.fit_transform(test)
training = pd.DataFrame(testing, columns= features)
test_X = pd.DataFrame(testing, columns=features)
test_Y = test.rlf
#
#
#
dimension = len(features)

lr = 1e-5
num_epochs = 100


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=dimension, out_features=int(dimension / 2))
        self.enc2 = nn.Linear(in_features=int(dimension / 2), out_features=int(dimension / 4))
        self.enc3 = nn.Linear(in_features=int(dimension / 4), out_features=int(dimension / 8))
        # self.enc4 = nn.Linear(in_features=int(dim/4), out_features=int(dim/8))

        # decoder
        self.dec1 = nn.Linear(in_features=int(dimension / 8), out_features=int(dimension / 4))
        self.dec2 = nn.Linear(in_features=int(dimension / 4), out_features=int(dimension / 2))
        self.dec3 = nn.Linear(in_features=int(dimension / 2), out_features=dimension)
        # self.dec4 = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x):
                x = F.relu(self.enc1(x))
                x = F.relu(self.enc2(x))
                x = F.relu(self.enc3(x))

                x = F.relu(self.dec1(x))
                x = F.relu(self.dec2(x))
                x = F.relu(self.dec3(x))

        # sigmoid activation
        # x = torch.sigmoid(self.enc1(x))
        # x = torch.sigmoid(self.enc2(x))
        # x = torch.sigmoid(self.enc3(x))
        # # x = F.relu(self.enc4(x))

        # x = torch.sigmoid(self.dec1(x))
        # x = torch.sigmoid(self.dec2(x))
        # x = torch.sigmoid(self.dec3(x))
                return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AutoEncoder()
optimizer = optim.Adam(net.parameters(), lr=1e-5)

loss_function = nn.BCEWithLogitsLoss()  # nn.BCEWithLogitsLoss()  #MSELoss too
get_loss = list()


def training_ae(net, trainloader, epochs):
    train_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            input_data = data.to(device=device)
            optimizer.zero_grad()
            output = net(input_data).to(device=device)  # output is the reconstruced x
            loss = loss_function(output, input_data).to(device=device)  # input_data should be the target variable
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)

        if epoch % 20 == 0:
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch + 1, num_epochs, loss))
    return train_loss


get_loss = training_ae(net, train_loader, num_epochs)

_, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.xlabel("epochs")
plt.ylabel("loss value ")
ax.set_title('Loss graph')
ax.plot(get_loss)

test_loss = []
net.eval()
test_tensor = torch.tensor(test_X.values.astype(np.float32))

with torch.no_grad():
    for i in range(len(test_X)):
        input = test_tensor[i].to(device=device)
        output = net(input).to(device=device)
        loss = loss_function(output, input).to(device=device)
        test_loss.append(loss.item())

fpr, tpr, thresholds = roc_curve(y_true=test_Y.astype(int), y_score=test_loss, pos_label=1)
ranked_thresholds = sorted(list(zip(np.abs(1.5*tpr - fpr), thresholds, tpr, fpr)), key=lambda i: i[0], reverse=True)
_, failure_threshold, threshold_tpr, threshold_fpr = ranked_thresholds[0]
print(f"Selected failure Threshold: {failure_threshold}")
print("Theshold yields TPR: {:.4f}, FPR: {:.4f}".format(threshold_tpr, threshold_fpr))

auc = roc_auc_score(y_true=test_Y.astype(int),  y_score=test_loss)
print("AUC: {:.4f}".format(auc))

plt.figure(figsize=(10, 10))
plt.plot([0,1], [0,1], linestyle="--") # plot baseline curve
plt.plot(fpr, tpr, marker=".", label="Failure Threshold:{:.6f}\nTPR: {:.4f}, FPR:{:.4f}".format(failure_threshold, threshold_tpr, threshold_fpr))
plt.axhline(y=threshold_tpr, color='darkgreen', lw=0.8, ls='--')
plt.axvline(x=threshold_fpr, color='darkgreen', lw=0.8, ls='--')
plt.title("ROC Curve")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc="lower right")

test_results = test_Y.to_frame().astype(bool)
test_results['loss'] = pd.Series(test_loss, index=test_results.index)
test_results['is_failed'] = test_results.loss > failure_threshold

conf_matrix = confusion_matrix(test_results.rlf, test_results.is_failed)
plt.figure()
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, fmt='g')
plt.title('Failure Threshold Classification - Confusion Matrix')
print(classification_report(test_results.rlf, test_results.is_failed, target_names=["regular", "rlf"]))