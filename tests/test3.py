import torch
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torchvision.transforms as transforms


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
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 512),
            nn.ReLU(),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x


model = OurCNN().to(device)



image_path = "./Otest.png"
model = OurCNN()
model.load_state_dict(torch.load('../models/X_O_recognition.pth',  map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

image = Image.open(image_path)
image = transform(image).unsqueeze(0)


with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)

predicted_letter = chr(predicted.item() + 96)
print(predicted_letter)