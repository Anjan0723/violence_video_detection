import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras import layers, models

FRAME_DIR = "frames"
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 64
EPOCHS = 10

def load_data():
    data = []
    labels = []

    for category in ['Violence', 'NonViolence']:
        folder = os.path.join(FRAME_DIR, category)
        label = 1 if category == 'Violence' else 0
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                frame = np.load(os.path.join(folder, file))
                data.append(frame)
                labels.append(label)

    return np.array(data), np.array(labels)

# Load dataset
X, y = load_data()
print(f"[INFO] Loaded {len(X)} samples.")

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
def create_model():
    model = models.Sequential([
        layers.Input(shape=IMAGE_SHAPE),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test), verbose=1)

# Save
model.save("violence_detection_model.h5")
print("[INFO] Model saved as 'violence_detection_model.h5'")

# Evaluate
y_pred = model.predict(x_test)
y_pred_label = (y_pred > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_label))

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()