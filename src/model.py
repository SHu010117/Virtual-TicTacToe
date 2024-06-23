import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

'''
batch_size = 64

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load the training and the test datasets
train_data = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True,
                                         transform=transform)
test_data = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True,
                                        transform=transform)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)




# get the best device for computation
device = ('cuda' if torch.cuda.is_available() else 'cpu')

'''

# Passo 3: Definire la struttura della CNN
class OurCNN(nn.Module):
    def __init__(self):
        super(OurCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12544, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,27)

        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x

'''

model = OurCNN().to(device)

# define the hyperparameters
epochs = 9
learning_rate = 0.001

# define the loss function
loss_fn = nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# define the accuracy metric
metric = torchmetrics.Accuracy(task='multiclass', num_classes=27).to(device)


# defining the training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # get the batch from the dataset
    for batch, (X, y) in enumerate(dataloader):

        # move data to device
        X = X.to(device)
        y = y.to(device)

        # compute the prediction and the loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # let's adjust the weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print some informations
        if batch % 1000 == 0:
            loss_v, current_batch = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss_v} [{current_batch}/{size}]')
            acc = metric(pred, y)
            print(f'Accuracy on current batch: {acc}')

    # print the final accuracy of the model
    acc = metric.compute()
    print(f'Final TRAINING Accuracy: {acc}')
    metric.reset()


# define the testing loop
def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # disable weights update
    with torch.no_grad():
        for X, y in dataloader:
            # move data to the correct device
            X = X.to(device)
            y = y.to(device)

            # get the model predictions
            pred = model(X)
            acc = metric(pred, y)



    # compute the final accuracy
    acc = metric.compute()
    print(f'FINAL TESTING ACCURACY: {acc}')
    metric.reset()


# train the model!!!
for epoch in range(epochs):
    print(f'-------------- Epoch: {epoch} --------------')
    print("Training...")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    print("Testing...")
    test_loop(test_dataloader, model)

print("Done!")
torch.save(model.state_dict(), "./OurCNN.pth")
'''
