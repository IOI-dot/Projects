import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]["Path"])
        label = int(self.data.iloc[idx]["ClassId"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ==== 1. Config ====
DATA_DIR = r"archive (11)"  # Train/ and Test/
BATCH_SIZE = 64
IMG_SIZE = 48
EPOCHS = 15
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== 2. Transforms ====
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# ==== 3. Load datasets ====
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "Train"), transform=train_transforms)

# Fix class_to_idx mapping (numeric)
train_data.class_to_idx = {k: int(k) for k in train_data.class_to_idx.keys()}

# Rebuild samples & targets using corrected mapping (Chatgpt) lol
new_samples = []
new_targets = []
for path, old_class in train_data.samples:
    class_name = os.path.basename(os.path.dirname(path))  # folder name
    new_class = train_data.class_to_idx[class_name]
    new_samples.append((path, new_class))
    new_targets.append(new_class)
train_data.samples = new_samples
train_data.targets = new_targets

print("Fixed class_to_idx mapping:")
print(train_data.class_to_idx)

val_data = GTSRBTestDataset(
    csv_file=os.path.join(DATA_DIR, "Test.csv"),
    root_dir=DATA_DIR,
    transform=val_transforms
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
print(f"Classes found: {len(train_data.classes)} total")
# ==== 4. Define CNN ====
class Net(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        reduced = img_size // 8
        flattened = 128 * reduced * reduced
        self.classifier = nn.Sequential(
            nn.Linear(flattened, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = Net(num_classes=len(train_data.classes), img_size=IMG_SIZE).to(device)

# ==== 5. Loss & Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==== 6. Training Loop ====
for epoch in range(EPOCHS):
    # Train
    model.train()
    correct = 0
    for i,(inputs, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print(f"Processing batch {i + 1}/{len(train_loader)}")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)
    train_acc = correct.double() / len(train_data)

    # Validate
    model.eval()
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_acc = correct.double() / len(val_data)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

# ==== 7. Confusion Matrix ====
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(len(train_data.classes))))
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
plt.title("GTSRB Confusion Matrix")
plt.tight_layout()
plt.savefig("gtsrb_confusion_matrix.png")
print("Confusion matrix saved as gtsrb_confusion_matrix.png")


# ==== 8. Save model ====
torch.save(model.state_dict(), "gtsrb_cnn.pth")
print("Model saved as gtsrb_cnn.pth")
