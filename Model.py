import torch
import numpy as np
from Spectogram import Spectogram
import os
class Model:

    def __init__(self, model, loader, optimizer, device, criterion):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion

    def train(self):
        total_loss = 0
        for _, batch in enumerate(loader, 0):

            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        return total_loss / len(loader)

    def test(self):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in self.loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output = model(inputs)
                output = output.to('cpu')
                labels = labels.to('cpu')
                for i in range(len(inputs)):
                    prediction = int(torch.argmax(output[i]))
                    true_label = int(labels[i])
                    if prediction == true_label:
                        correct += 1
                    total += 1
        return correct / total

class Spectogram_DataLoader(torch.utils.data.DataLoader):
    def __init__(self, spectogram_files):
        self.length = len(spectogram_files)
        self.spectograms = []
        for spectogram_file, label in spectogram_files:
            print("Loading " + spectogram_file)
            spectogram = Spectogram.load_spectogram_from_file(spectogram_file)
            #Arguably spectogram in db scale is a better image 
            spectogram = Spectogram.real_spectogram_to_db_spectogram(spectogram)
            spectogram = torch.from_numpy(spectogram)
            self.spectograms.append((spectogram, label))
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.spectograms[idx]
        
class FC(torch.nn.Module):
    def __init__(self, activation, num_hidden_nodes, num_inputs, outputs):
        super().__init__()
        num_hidden_nodes.insert(0, num_inputs)
        num_hidden_nodes.append(outputs)
        self.layers = torch.nn.ModuleList()
        for i in range(len(num_hidden_nodes) - 1):
            self.layers.append(torch.nn.Linear(num_hidden_nodes[i], num_hidden_nodes[i+1]))
        self.activation = activation

    def forward(self, x, num_inputs):
        x= x.view(-1, num_inputs)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

def loader(loader_type):
    files = []
    prefix = None
    if loader_type == 'train':
        prefix = 'training_data/'

    elif loader_type == 'test':
        prefix = 'test_data'

    else:
        return None

    label_dirs = os.listdir(prefix)
    for label_dir in label_dirs:
        if 'spec' not in label_dir:
            continue
        path = prefix + label_dir + '/'
        for wav_file in os.listdir(path):
            files.append(spectogram_file, int(prefix[4:]))

        return Spectogram_DataLoader(files)

        

loader('train')