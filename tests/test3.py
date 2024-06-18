import torch
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


# get the best device for computation
device = ('cuda' if torch.cuda.is_available() else 'cpu')


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
            nn.Linear(12544, 800),
            nn.ReLU(),
            nn.Linear(800, 120),
            nn.ReLU(),
            nn.Linear(120, 62)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x


model = OurCNN().to(device)



image_path = "./Otest.png"
model.load_state_dict(torch.load('../models/X_O_CNN.pth',  map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Assicurati che l'immagine sia 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = Image.open(image_path)
image = transform(image).unsqueeze(0)


image = image.to(device)

with torch.no_grad():
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    _, predicted = torch.max(output, 1)
    probabilities = probabilities.cpu().numpy()[0]



class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 0-9
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # A-Z
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'   # a-z
]

for i, prob in enumerate(probabilities):
    print(f"Class {i} ({class_names[i]}): {prob:.4f}")

'''

label = predicted.item()
probability_O = probabilities[0][0].item()
probability_X = probabilities[0][1].item()

if label == 0:
    print(f"La lettera è O con una probabilità del {probability_O * 100:.2f}%")
    print(f"Probabilità di X: {probability_X * 100:.2f}%")
else:
    print(f"La lettera è X con una probabilità del {probability_X * 100:.2f}%")
    print(f"Probabilità di O: {probability_O * 100:.2f}%")
    
'''
