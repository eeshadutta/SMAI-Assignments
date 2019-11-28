import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
print(torch.cuda.is_available())
use_cuda = True

input_size = 784
hidden_size = [1000, 1000]
output_size = 10
epochs = 10
batch_size = 100
lr = 1e-2

train_dataset = dsets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False,
                           transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_method):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], output_size)
        self.nl = nn.ReLU()
        self.layers = nn.Sequential(
            self.l1, self.nl, self.l2, self.nl, self.l3)
        if init_method == 'uniform':
            nn.init.uniform_(self.l1.weight)
            nn.init.uniform_(self.l2.weight)
            nn.init.uniform_(self.l3.weight)
        elif init_method == 'gaussian':
            nn.init.normal_(self.l1.weight)
            nn.init.normal_(self.l2.weight)
            nn.init.normal_(self.l3.weight)
        else:
            nn.init.xavier_normal_(self.l1.weight)
            nn.init.xavier_normal_(self.l2.weight)
            nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):
        out = self.layers(x)
        return out


def training(epochs, input_size, hidden_size, output_size, init_method, train_loader, lr):
    mlp = MLP(input_size, hidden_size, output_size, init_method)
    if use_cuda and torch.cuda.is_available():
        mlp.cuda()
    print('Using', init_method, 'initialisation and learning rate', lr)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr)
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = mlp(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], step = [%d/%d], Loss = %.5f' %
                      (epoch+1, epochs, i+1, len(train_dataset)//batch_size, loss.data))
        epoch_loss = running_loss/len(train_dataset)
        losses.append(epoch_loss)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))

        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = mlp(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Acc = %.4f%%' % (100 * correct / total))
    return losses


lr = [10, 1, 1e-1, 1e-2, 1e-4, 1e-5]
initialisation = ['xavier', 'uniform', 'gaussian']
for inits in initialisation:
    plt.figure(figsize=[15, 15], dpi=60)
    i = 1
    for learning_rate in lr:
        plt.subplot(2, 3, i)
        losses = training(epochs, input_size, hidden_size,
                          output_size, inits, train_loader, learning_rate)
        plt.plot(losses, label='Loss Curve\n Initialisation = ' +
                 inits+'\n Learning Rate = '+str(lr))
        plt.title('Learning Rate = ' + str(learning_rate), fontsize=25)
        i += 1
    plt.suptitle('Loss Curve\n Initialisation = '+inits, fontsize=25)
    plt.savefig('1_'+inits+'.png')
    plt.show()
