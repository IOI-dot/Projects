import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

Data_Folder = r"C:\Users\Omar\Downloads\Emotions"
Image_Size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)
model.load_state_dict(torch.load("emotions_resnet.pth", map_location=device))
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((Image_Size, Image_Size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

classes = ['Sad','Surprised','Happy', 'Neutral','Sad','Angry']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            print(pred.item())
            emotion = classes[pred.item()]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
