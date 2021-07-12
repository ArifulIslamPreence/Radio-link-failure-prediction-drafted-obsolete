'''
Implementation of Generative Adverserial Network for dataset balancing.
 The whole combined dataset is fed into model by spliting batches
'''
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import math
import matplotlib.pyplot as plt

df1 = pd.read_csv("../output_dataset/combined_short-main-2.csv", index_col=0, low_memory=False)
train, test = train_test_split(df1, test_size=0.30, random_state=0)
features = train.columns
batch_size = 100
training = preprocessing.normalize(train)
train_data = pd.DataFrame(training, columns=features)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

dimension = len(features)
# Calculator selection

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# GAN implementation

# NN for discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # setting 3 layers
            nn.Linear(dimension, dimension / 2),  # confusion about setting dimension
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dimension / 2, dimension / 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dimension / 4, dimension / 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dimension / 8, dimension / 16),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


# NN for fake data generation
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(

            # 3 layers
            nn.Linear(dimension / 16, dimension / 8),
            nn.ReLU(),
            nn.Linear(dimension / 8, dimension / 4),
            nn.ReLU(),
            nn.Linear(dimension / 4, dimension / 2),
            nn.ReLU(),
            nn.Linear(dimension / 2, dimension),
        )

    def forward(self, x):
        output = self.model(x)
        return output


generator = Generator().to(device=device)
discriminator = Discriminator().to(device=device)

# Model Training

lr = 1e-5
num_epochs = 50
loss_function = nn.BCEWithLogitsLoss()

# Model Optimizer

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

# Start trainings


for epoch in range(num_epochs):
    lossGen = []
    lossDis = []
    for n, (real_samples, features) in enumerate(train_loader):

        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(
            device=device)  # create labels with the value 1 for the real samples. Do we need labeling for supervised
        # data?
        latent_space_samples = torch.randn((batch_size, 2)).to(
            device=device)  # assign the 2 as labels to real_samples_labels.

        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(
            device=device)  # create labels with the value 1 for the real samples
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator,
                                           all_samples_labels)  # calculate the loss function using the output from the model
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2)).to(device=device)

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)  # loss function
        loss_generator.backward()
        optimizer_generator.step()
        lossGen.append(loss_generator)
        lossDis.append(loss_discriminator)
        # Show loss for each 5 epoch
        if epoch % 5 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
_, ax = plt.subplots(1, 1, figsize=(15, 10))
plt.xlabel("epochs")
plt.ylabel("Generator loss value ")
ax.set_title(' Generator Loss graph')
ax.plot(lossGen)

_, bx = plt.subplots(1, 1, figsize=(15, 10))
plt.xlabel("epochs")
plt.ylabel("Discriminator loss value ")
bx.set_title('Discriminator  Loss graph')
bx.plot(lossDis)
