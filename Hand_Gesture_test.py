import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import json

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
with open("classes.json", "r") as f:
    CLASSES = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load trained gesture model ====
model = Net(num_classes=len(CLASSES), img_size=IMG_SIZE).to(device)
model.load_state_dict(torch.load("hand_gesture_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ==== MediaPipe Hands ====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box of the hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Add a margin around the box
            margin = 20
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Convert to tensor
            pil_img = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Predict gesture
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
                conf, pred = torch.max(probs, 1)  # Get confidence and predicted class
                conf = conf.item()
                label = CLASSES[pred.item()]

            # Only show if confidence is high
            CONF_THRESHOLD = 0.8
            if conf >= CONF_THRESHOLD:
                # Draw results
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
