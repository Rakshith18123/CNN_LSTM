import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from extract_features import load_dataset  # Importing from extract_features.py

# Path to the CREMA-D dataset's audio files
dataset_path = r'C:\Users\raksh\CREMA-D\AudioWAV'

# Load and preprocess the dataset
features, labels = load_dataset(dataset_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Reshape features to add the channel dimension
features = features.reshape(features.shape[0], 40, 40, 1)

# Load the trained model
model = load_model('cnn_lstm_emotion_recognition_model.keras')

# Evaluate the model on the dataset
loss, accuracy = model.evaluate(features, labels_onehot)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predict emotions for the dataset
predictions = model.predict(features)
predicted_emotions = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Compare with actual labels
actual_emotions = label_encoder.inverse_transform(np.argmax(labels_onehot, axis=1))

# Print some sample predictions
for i in range(10):
    print(f"Actual Emotion: {actual_emotions[i]}, Predicted Emotion: {predicted_emotions[i]}")

