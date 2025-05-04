import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
DATASET_PATH = 'asl_alphabet_train/asl_alphabet_train'

IMG_SIZE = 64

def load_data():
    images = []
    labels = []
    classes = sorted(os.listdir(DATASET_PATH))
    for idx, label in enumerate(classes):
        folder_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(folder_path):
            continue
        for img_file in os.listdir(folder_path)[:1000]:  # Limit to 1000 per class for faster training
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)
    return np.array(images), to_categorical(labels), classes

print("[INFO] Loading data...")
X, y, label_names = load_data()
X = X / 255.0

print("[INFO] Splitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Building model...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

model.save("asl_model.h5")
print("[✅] Model trained and saved as asl_model.h5")
