import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# ==== Load Model ====
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

IMG_SIZE = 64
CLASSES = ["happy", "neutral"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net(num_classes=len(CLASSES), img_size=IMG_SIZE).to(device)
model.load_state_dict(torch.load("happy_neutral_cnn.pth", map_location=device))
model.eval()

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==== Face Detector ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==== Webcam ====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_pil = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_tensor = transform(transforms.ToPILImage()(face_pil)).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            _, pred = torch.max(outputs, 1)
            label = CLASSES[pred.item()]

        color = (0, 255, 0) if label == "happy" else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Happy / Neutral Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
