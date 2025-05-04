import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('asl_model.h5')
labels = [chr(i) for i in range(65, 91)]  # A–Z

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
IMG_SIZE = 64

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            x_min, y_min = int(x_min) - 20, int(y_min) - 20
            x_max, y_max = int(x_max) + 20, int(y_max) + 20

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                try:
                    resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    normalized = resized / 255.0
                    reshaped = np.expand_dims(normalized, axis=0)
                    pred = model.predict(reshaped)
                    class_id = np.argmax(pred)
                    confidence = np.max(pred)

                    label = f"{labels[class_id]} ({confidence:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except:
                    pass

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
