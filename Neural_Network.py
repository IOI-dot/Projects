import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# ==== 1. Config ====
DATA_DIR = r"C:\Users\Omar\Downloads\animals_3class"  # should have "train" and "valid" folders
BATCH_SIZE = 16
IMG_SIZE = 64
EPOCHS = 5
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 2. Transforms ====
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==== 3. Load datasets ====
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Classes: {train_data.classes}")

# ==== 4. Define a simple CNN ====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=len(train_data.classes)).to(device)

# ==== 5. Loss & Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==== 6. Training loop ====
for epoch in range(EPOCHS):
    model.train()
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)

    train_acc = running_corrects.double() / len(train_data)

    # Validation
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels)

    val_acc = val_corrects.double() / len(val_data)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

# ==== 7. Save model ====
torch.save(model.state_dict(), "bat_bear_bee_cnn.pth")
print("✅ Model saved!")

# ==== 8. Prediction function ====
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = val_transforms(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return train_data.classes[pred.item()]

# Example:
print("Prediction:", predict_image(r"C:\Users\Omar\Downloads\archive (4)\animals\animals\bear\.jpg"))
