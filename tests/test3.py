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
            nn.Linear(12544, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 27)

        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x


model = OurCNN().to(device)

image_path = "./test_img/Screenshot 2024-06-19 alle 14.01.47.png"
model.load_state_dict(torch.load('../models/OurCNN2.pth', map_location=torch.device('cpu')))
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

with torch.no_grad():  # Disabilita il calcolo dei gradienti
    output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_prob, top_class = probabilities.topk(1, dim=1)
    probabilities = probabilities.squeeze().cpu().numpy()
    prediction = top_class.item()


letters = [chr(i + 96) for i in range(1, 27)]
probabilities_dict = {letters[i-1]: probabilities[i] for i in range(1, 27)}
print("\nProbabilità per ogni lettera:")
for letter, prob in probabilities_dict.items():
    print(f'{letter}: {prob:.4f}')


