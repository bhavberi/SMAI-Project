import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import pandas as pd
from random import shuffle
import os
import librosa
import numpy as np

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 500
num_examples = 8000
num_classes = 4
alpha = 0.0001
num_epochs = 200
batch_size = 512
classification_weight = 0.1

def get_one_hot(label_num, num_classes=4):
    one_hot = np.zeros((1, num_classes))
    one_hot[0, int(label_num)] = 1
    return one_hot

def load_data():
    print('Reading data...')
    songs = np.zeros((num_examples, input_dim))
    one_hot_labels = np.zeros((num_examples, num_classes))
    counter = 0

    all_genres = ['classical', 'jazz', 'metal', 'pop']
    num_split = 20
    size_split = input_dim

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

                for j in range(num_split):
                    songs[counter] = audio[(size_split * j):(size_split * (j + 1))]
                    one_hot_labels[counter] = get_one_hot(index)
                    counter += 1

    songs = pd.DataFrame(songs)
    one_hot_labels = pd.DataFrame(one_hot_labels)
    print('Data reading done :)')
    return songs, one_hot_labels

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AutoencoderClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 192),
            nn.Tanh(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 192),
            nn.Sigmoid(),
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        classification = self.classifier(encoding)
        return encoding, decoding, classification

def train(model, criterion, optimizer, dataloader, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        cost_list = []
        curr_num_correct = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            encoding, decoding, preds = model(inputs)
            reconstruction_loss = criterion(decoding, inputs)
            classification_loss = classification_weight * criterion(preds, labels.float())  # Convert labels to float
            loss = reconstruction_loss + classification_loss
            loss.backward()
            optimizer.step()

            _, predictions = torch.max(preds, 1)
            _, labels = torch.max(labels, 1)
            curr_num_correct += torch.sum(predictions == labels).item()

            cost_list.append(loss.item())

        accuracy = curr_num_correct / float(len(dataloader.dataset))
        print(f"Epoch {epoch + 1}, Train Accuracy: {accuracy}")
        loss_per_epoch = float(sum(cost_list)) / len(cost_list)
        print(f"Epoch {epoch + 1}, Train Loss: {loss_per_epoch}")

def test(model, dataloader):
    model.eval()
    correct_class = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, preds = model(inputs)

            _, predictions = torch.max(preds, 1)
            _, labels = torch.max(labels, 1)
            correct_class += torch.sum(predictions == labels).item()

    accuracy = correct_class / float(len(dataloader.dataset))
    print(f"Test Accuracy: {accuracy}")

def main():
    songs, labels = load_data()

    ind_list = [i for i in range(songs.shape[0])]
    shuffle(ind_list)
    songs = songs.iloc[ind_list]
    labels = labels.iloc[ind_list]

    songs_train = songs.iloc[0:6000]
    songs_dev = songs.iloc[6000:]
    labels_train = labels.iloc[0:6000]
    labels_dev = labels.iloc[6000:]

    dataset_train = data.TensorDataset(torch.from_numpy(songs_train.values).float(), torch.from_numpy(labels_train.values).long())
    dataset_dev = data.TensorDataset(torch.from_numpy(songs_dev.values).float(), torch.from_numpy(labels_dev.values).long())

    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_dev = data.DataLoader(dataset_dev, batch_size=batch_size, shuffle=False)

    model = AutoencoderClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    train(model, criterion, optimizer, dataloader_train)
    test(model, dataloader_dev)

if __name__ == "__main__":
    main()

