import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil

# ==== 1. Config ====
DATA_DIR = r"C:\Users\Omar\Downloads\Emotions"  # Has train/ and test/ each with 7 emotions
BATCH_SIZE = 16
IMG_SIZE = 64
EPOCHS = 15
LR = 0.001
TARGET_CLASSES = ["happy", "neutral"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== 2. Filter dataset to only the picked stuff====
def filter_dataset(original_dir, filtered_dir, target_classes):
    # Delete old filtered dataset if it exists
    if os.path.exists(filtered_dir):
        shutil.rmtree(filtered_dir)

    for split in ["train", "test"]:
        for cls in target_classes:
            src = os.path.join(original_dir, split, cls)
            dst = os.path.join(filtered_dir, split, cls)

            if not os.path.exists(src):
                print(f"WARNING: Source folder not found: {src}")
                continue

            os.makedirs(dst, exist_ok=True)

            for file in os.listdir(src):
                src_file = os.path.join(src, file)
                if os.path.isfile(src_file):  # avoid copying dirs
                    shutil.copy(src_file, dst)

FILTERED_DIR = os.path.join(os.path.dirname(DATA_DIR), "emotions_happy_neutral")
filter_dataset(DATA_DIR, FILTERED_DIR, TARGET_CLASSES)

# ==== 3. Transforms ====
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==== 4. Load datasets ====
train_data = datasets.ImageFolder(os.path.join(FILTERED_DIR, "train"), transform=train_transforms)
val_data = datasets.ImageFolder(os.path.join(FILTERED_DIR, "test"), transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes found: {train_data.classes}")

# ==== 5. Define CNN ====
class Net(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
        )
        reduced_size = img_size // 4
        flattened_features = 64 * reduced_size * reduced_size
        self.classifier = nn.Sequential(
            nn.Linear(flattened_features, 128),
            nn.ELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

model = Net(num_classes=len(train_data.classes), img_size=IMG_SIZE).to(device)

# ==== 6. Loss & Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==== 7. Training Loop ====
for epoch in range(EPOCHS):
    model.train()
    running_corrects = 0
    for i, (inputs, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print(f"Processing batch {i + 1}/{len(train_loader)}")
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

# ==== 8. Save model ====
torch.save(model.state_dict(), "happy_neutral_cnn.pth")
print("Model saved as happy_neutral_cnn.pth")
