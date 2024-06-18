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

batch_size = 32

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load the training and the test datasets
train_data = torchvision.datasets.EMNIST(root='../data', split='letters', train=True, download=True,
                                         transform=transform)
test_data = torchvision.datasets.EMNIST(root='../data', split='letters', train=False, download=True,
                                        transform=transform)


class_labels = {'X': 24, 'O': 15}


def filter_dataset(dataset, labels):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels.values()]
    return Subset(dataset, indices)


def new_labels(dataset):
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == 15:
            dataset.dataset.targets[dataset.indices[i]] = 0
        elif label == 24:
            dataset.dataset.targets[dataset.indices[i]] = 1


train_data = filter_dataset(train_data, class_labels)
test_data = filter_dataset(test_data, class_labels)

new_labels(train_data)
new_labels(test_data)

'''
def show_images(dataset, num_images=6):
    figure = plt.figure(figsize=(10, 5))
    cols, rows = 3, 2
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title('X' if label == 1 else 'O')
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# Visualizzare 6 immagini dal subset
show_images(train_data, num_images=6)
'''

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# get the best device for computation
device = ('cuda' if torch.cuda.is_available() else 'cpu')


# Passo 3: Definire la struttura della CNN
class OurCNN(nn.Module):
    def __init__(self):
        super(OurCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12544, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x


model = OurCNN().to(device)

# define the hyperparameters
epochs = 3
learning_rate = 0.001

# define the loss function
loss_fn = nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# define the accuracy metric
metric = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)


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
        if batch % 100 == 0:
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