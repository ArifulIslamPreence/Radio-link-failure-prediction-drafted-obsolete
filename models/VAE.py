''' Implementation of Variational Autoencoder Network for dataset reconstructing into normalized form.
 The whole combined dataset is fed into model by spliting batches '''

import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
import math
import matplotlib.pyplot as plt

df1 = pd.read_csv("combined_short-main-2.csv", index_col=0, low_memory=False)
train, test = train_test_split(df1, test_size=0.30, random_state=0)
features = train.columns
batch_size = 100

X = StandardScaler().fit_transform(train)
X = np.reshape(X,(X.size,1))

# if np.any(np.isnan(X)):
#     np.where(np.isnan(X))
#     np.nan_to_num(X)
X = X[np.logical_not(np.isnan(X))]

print(np.any(np.isnan(X)))
training = preprocessing.normalize(X)
train_data = pd.DataFrame(training, columns=features)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

dimension = len(features)

lr = 1e-5
num_epochs = 50


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
        #         x = F.relu(self.enc1(x))
        #         x = F.relu(self.enc2(x))
        #         x = F.relu(self.enc3(x))

        #         x = F.relu(self.dec1(x))
        #         x = F.relu(self.dec2(x))
        #         x = F.relu(self.dec3(x))
        # sigmoid activation
        x = torch.sigmoid(self.enc1(x))
        x = torch.sigmoid(self.enc2(x))
        x = torch.sigmoid(self.enc3(x))
        # x = F.relu(self.enc4(x))

        x = torch.sigmoid(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        # x = F.relu(self.dec4(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AutoEncoder()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

loss_function = nn.BCELoss()  # nn.BCEWithLogitsLoss()  #MSELoss too
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

        if epoch % 5 == 0:
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch + 1, num_epochs, loss))
    return train_loss


get_loss = training_ae(net, train_loader, num_epochs)

_, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.xlabel("epochs")
plt.ylabel("loss value ")
ax.set_title('Loss graph')
ax.plot(get_loss)
