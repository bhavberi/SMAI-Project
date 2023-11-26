import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
from random import shuffle
import librosa
import os

input_dim = 500
num_classes = 4
alpha = 0.0001
num_epochs = 200
batch_size = 512

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

def get_one_hot(label_num, num_classes=4):
    one_hot = np.zeros((1, num_classes))
    one_hot[0, int(label_num)] = 1
    return one_hot

def load_data():
    print('Reading data...')
    songs = np.zeros((8000, input_dim))
    onehotlabels = np.zeros((8000, num_classes))
    counter = 0

    all_genres = ['classical', 'jazz', 'metal', 'pop']

    numsplit = 20
    sizesplit = input_dim

    for index in range(len(all_genres)):
        for filename in os.listdir('./Data/genres_original/' + all_genres[index]):
            if filename.endswith(".wav"):
                try:
                    audio, _ = librosa.load('./Data/genres_original/' + all_genres[index] + '/' + filename)
                except Exception as e:
                    print(f'Error encountered: {e}')
                    continue

                audio = audio[:600000]
                audio = audio.reshape(15000, 40)
                audio = np.mean(audio, axis=1)

                for j in range(numsplit):
                    songs[counter] = audio[(sizesplit * j): (sizesplit * (j + 1))]
                    onehotlabels[counter] = get_one_hot(index)
                    counter += 1

    songs = pd.DataFrame(songs)
    onehotlabels = pd.DataFrame(onehotlabels)
    print('Data reading done :)')
    return songs, onehotlabels

def train(model, train_loader, dev_loader, criterion, optimizer, num_epochs):
    model.train()
    
    train_accuracies = []
    dev_accuracies = []
    loss_per_epoch = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct_train = 0

        for inputs_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            loss = criterion(outputs, labels_batch.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, labels_batch = torch.max(labels_batch.data, 1)
            correct_train += (predicted == labels_batch).sum().item()

        accuracy_train = correct_train / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Accuracy: {accuracy_train}")

        model.eval()
        correct_dev = 0

        with torch.no_grad():
            for inputs_batch, labels_batch in dev_loader:
                outputs = model(inputs_batch)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_batch = torch.max(labels_batch.data, 1)
                correct_dev += (predicted == labels_batch).sum().item()

        accuracy_dev = correct_dev / len(dev_loader.dataset)
        print(f"Epoch {epoch+1}, Dev Accuracy: {accuracy_dev}")

        train_accuracies.append(accuracy_train)
        dev_accuracies.append(accuracy_dev)
        loss_per_epoch.append(total_loss / len(train_loader))

    return train_accuracies, dev_accuracies, loss_per_epoch

def main():
    songs, labels = load_data()

    # Shuffling training set
    ind_list = [i for i in range(songs.shape[0])]
    shuffle(ind_list)
    songs = songs.iloc[ind_list]
    labels = labels.iloc[ind_list]

    songs_train = songs.iloc[0:6000].values
    songs_dev = songs.iloc[6000:].values
    labels_train = labels.iloc[0:6000].values
    labels_dev = labels.iloc[6000:].values

    train_dataset = data.TensorDataset(torch.from_numpy(songs_train).float(), torch.from_numpy(labels_train).long())
    dev_dataset = data.TensorDataset(torch.from_numpy(songs_dev).float(), torch.from_numpy(labels_dev).long())

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = SoftmaxClassifier(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    train_accuracies, dev_accuracies, loss_per_epoch = train(model, train_loader, dev_loader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()

