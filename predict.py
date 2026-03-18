import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Class names
classes = ['early_blight','healthy','late_blight']

# Same model structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


# Load model
model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load image
img = Image.open("test_leaf.jpg")
img = transform(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output,1)

print("Prediction:", classes[predicted.item()])