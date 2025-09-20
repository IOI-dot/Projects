import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch
from sklearn.utils.class_weight import compute_class_weight
Data_Folder= r"C:\Users\Omar\Downloads\Emotions"
Batch_Size = 16
Image_Size = 128
EPOCHS = 20
LR = 0.001
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Training on CPU")
train_transform = transforms.Compose([transforms.Resize((Image_Size,Image_Size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ColorJitter(brightness=0.2,contrast=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],[0.5,0.5,0.5])])
valid_transform = transforms.Compose([transforms.Resize((Image_Size,Image_Size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],[0.5,0.5,0.5])])
train_dataset = datasets.ImageFolder(os.path.join(Data_Folder,"train"), transform=train_transform)
valid_dataset = datasets.ImageFolder(os.path.join(Data_Folder,"test"), transform=valid_transform)
train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=Batch_Size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import Counter
print("Train:", Counter(train_dataset.targets))
print("Val:", Counter(valid_dataset.targets))
print(train_dataset.class_to_idx)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_dataset.targets),
    y=train_dataset.targets
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
for epoch in range(EPOCHS):
    model.train()
    running_corrects = 0
    for i, (inputs, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print(f"Batch {i + 1}/{len(train_loader)}")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    train_acc = running_corrects.double() / len(train_dataset)
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        val_acc = running_corrects.double() / len(valid_dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
    scheduler.step(val_acc)
torch.save(model.state_dict(), "emotions_resnet.pth")
