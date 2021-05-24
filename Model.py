import torch

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

