import os
import librosa
import numpy as np

# Function to extract features from audio files
def extract_features(file_name):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_name, sr=None, res_type='kaiser_fast')  # Keep the original sample rate
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Pad or truncate MFCCs to ensure they are 40x40
        if mfccs.shape[1] < 40:
            mfccs = np.pad(mfccs, ((0, 0), (0, 40 - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :40]
        # Return MFCC features as 2D array
        return mfccs
    except Exception as e:
        # Print the error and file name for debugging
        print(f"Error encountered while parsing file: {file_name}")
        print(f"Exception: {e}")
        return None

# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    features = []
    labels = []

    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            file_path = os.path.join(dataset_path, file)
            try:
                # Extract emotion label from filename
                emotion = file.split('_')[2]  # Assuming the emotion label is the third component of the filename
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
                else:
                    print(f"Feature extraction failed for file: {file}")
            except Exception as e:
                print(f"Error while processing file: {file}")
                print(f"Exception: {e}")

    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Print some statistics about the dataset
    print(f"Loaded {len(features)} samples.")
    print(f"Feature shape: {features.shape}")
    print(f"Unique emotions: {np.unique(labels)}")

    return features, labels
