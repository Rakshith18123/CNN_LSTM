import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from extract_features import load_dataset
from build_model import build_cnn_lstm_model  # Import the CNN-LSTM model

# Path to the CREMA-D dataset's audio files
dataset_path = r'C:\Users\raksh\CREMA-D\AudioWAV'

# Load and preprocess the dataset
features, labels = load_dataset(dataset_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Reshape features to (num_samples, 40, 40, 1) for CNN-LSTM model
features = features.reshape(features.shape[0], 40, 40, 1)

# Check the shape of the features
print(f"Feature shape: {features.shape}")

# Build and compile the model
num_classes = len(np.unique(labels))  # Number of unique emotions
model = build_cnn_lstm_model(input_shape=(40, 40, 1), num_classes=num_classes)  # Update to match the input shape

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the entire dataset
model.fit(features, labels_onehot, epochs=50, batch_size=32)

# Evaluate the model on the same dataset
loss, accuracy = model.evaluate(features, labels_onehot)
print(f"Training Accuracy: {accuracy*100:.2f}%")

# Save the model
model.save('cnn_lstm_emotion_recognition_model.keras')
print("Model trained and saved to disk.")
