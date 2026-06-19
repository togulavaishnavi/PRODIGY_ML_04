import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ======================
# CONFIG
# ======================
data_path = "leapGestRecog"
img_size = 64

X, Y = [], []

print("🚀 Loading dataset...")

# ======================
# LOAD DATA
# ======================
for root, dirs, files in os.walk(data_path):

    if len(files) > 0:
        label = os.path.basename(root)

        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):

                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    X.append(img)
                    Y.append(label)

print("✅ Images loaded:", len(X))

if len(X) == 0:
    print("❌ Dataset not loaded. Check folder path.")
    exit()

# ======================
# PREPROCESS
# ======================
X = np.array(X).reshape(-1, 64, 64, 1) / 255.0

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = to_categorical(Y)

# ======================
# SPLIT DATA
# ======================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("🧠 Data ready")

# ======================
# MODEL
# ======================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(Y.shape[1], activation='softmax')
])

# ======================
# COMPILE
# ======================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ======================
# TRAIN
# ======================
print("🔥 Training started...")

model.fit(X_train, Y_train,
          epochs=5,
          batch_size=32,
          validation_data=(X_test, Y_test))

# ======================
# SAVE MODEL
# ======================
model.save("gesture_model.h5")

print("🎉 Task 4 Completed Successfully!")