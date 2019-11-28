import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init as weight_init
import matplotlib.pyplot as plt
import numpy as np
import pdb


# parameters
batch_size = 128

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Loading the train set file
dataset = datasets.MNIST(root='./data', transform=preprocess, download=True)

loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True)


class AE(nn.Module):
    def __init__(self, num_neurons):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_neurons),
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_neurons, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.encoder(x)
        xr = self.decoder(h)
        return xr, h


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using CUDA ', use_cuda)

neurons = [2, 4, 8, 16, 32, 64]


def neural_net(opt):
    print('Optimizer:', opt)
    reconstruction_errors = []
    for x in neurons:
        print('Number of hidden neurons', x)
        net = AE(x)
        net = net.to(device)

        # Mean square loss function
        criterion = nn.MSELoss()

        # Parameters
        learning_rate = 1e-2
        weight_decay = 1e-5

        #Optimizer and Scheduler
        if opt == 'SGD without momentum':
            optimizer = torch.optim.SGD(
                net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.1)
        elif opt == 'SGD with momentum':
            optimizer = torch.optim.SGD(
                net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0)
        elif opt == 'Adam':
            optimizer = torch.optim.Adam(
                net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.RMSprop(
                net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, threshold=0.001, patience=5, verbose=True)
        num_epochs = 10

        # Training
        epochLoss = []
        for epoch in range(num_epochs):
            total_loss, cntr = 0, 0

            for i, (images, _) in enumerate(loader):
                images = images.view(-1, 28*28)
                images = images.to(device)

                # Initialize gradients to 0
                optimizer.zero_grad()

                # Forward pass (this calls the "forward" function within Net)
                outputs, _ = net(images)

                # Find the loss
                loss = criterion(outputs, images)

                # Find the gradients of all weights using the loss
                loss.backward()

                # Update the weights using the optimizer and scheduler
                optimizer.step()

                total_loss += loss.item()
                cntr += 1

            scheduler.step(total_loss/cntr)
            print('Epoch [%d/%d], Loss: %.4f' %
                  (epoch+1, num_epochs, total_loss/cntr))
            epochLoss.append(total_loss/cntr)
        reconstruction_errors.append(epochLoss[len(epochLoss) - 1])

    return reconstruction_errors


optimizers = ['SGD without momentum', 'SGD with momentum', 'Adam', 'RMSProp']
for opt in optimizers:
    reconstruction_errors = neural_net(opt)
    plt.plot(neurons, reconstruction_errors)
    plt.xlabel('Hidden Neurons')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs Number of Hidden Neurons\nOptimizer :' + opt)
    plt.savefig('2_'+opt+'.png')
    plt.show()
