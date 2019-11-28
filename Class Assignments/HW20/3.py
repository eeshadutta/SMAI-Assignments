import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init as weight_init
import matplotlib.pyplot as plt
import pdb
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Loading the train set file
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=preprocess,
                               download=True)
# Loading the test set file
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=preprocess)

# Dataloader
batch_size = 100

# loading the train dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# loading the test dataset
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Parameters
input_size = 784
hidden_size = [1000, 1000]
output_size = 10
num_classes = 10
num_epochs = 10
learning_rate = 0.01
momentum_rate = 0.9


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size[1], output_size)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weight_init.xavier_normal_(m.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def training_MLP():
    mlp = MLP(input_size, hidden_size, num_classes)
    if use_cuda and torch.cuda.is_available():
        mlp.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    losses = []
    output_store = {}
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            labels = labels
            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = mlp(images)
            output_store[labels] = outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], step = [%d/%d], Loss = %.5f' % (epoch+1,
                                                                      num_epochs, i+1, len(train_dataset)//batch_size, loss.data))
        epoch_loss = running_loss/len(train_dataset)
        losses.append(epoch_loss)

    return output_store, mlp


def training_MLP_enc(model):
    mlp = MLP(input_size, hidden_size, num_classes)
    if use_cuda and torch.cuda.is_available():
        mlp.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    losses = []
    output_store = {}
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            labels = labels
            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = mlp(model(images))
            output_store[labels] = outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], step = [%d/%d], Loss = %.5f' % (epoch+1,
                                                                      num_epochs, i+1, len(train_dataset)//batch_size, loss.data))
        epoch_loss = running_loss/len(train_dataset)
        losses.append(epoch_loss)

    return output_store, mlp


def testing_MLP(model):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28*28)

    if use_cuda and torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    acc = 100 * correct / total
    print('Acc = %.4f%%' % (acc))
    return acc


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


def training_AE():
    ae = Autoencoder()
    ae = ae.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), learning_rate)
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1)
            labels = labels
            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs_enc = ae(images)
            loss = criterion(outputs_enc, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], step = [%d/%d], Loss = %.5f' % (epoch+1,
                                                                      num_epochs, i+1, len(train_dataset)//batch_size, loss.data))
        epoch_loss = running_loss/len(train_dataset)
        losses.append(epoch_loss)

    return ae


output_raw, mlp_raw = training_MLP()

ae = training_AE()

output_enc, mlp_enc = training_MLP_enc(ae)

acc_raw = testing_MLP(mlp_raw)

acc_enc = testing_MLP(mlp_enc)

label = ['raw pixel features', 'hidden features']
acc = [acc_raw, acc_enc]
index = np.arange(len(label))
plt.figure(figsize=[10, 10], dpi=60)
plt.bar(index, acc, width=0.5)
plt.xlabel('Method', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title("Performance Comparison", fontsize=30)
plt.savefig('3_plot.png')
plt.show()
