import os
import librosa
import numpy as np
import multiprocessing
import time
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Function to extract audio features
def extract_features(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate), axis=1)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate), axis=1)

    return np.concatenate((chroma, spectral_contrast, mfcc))


def process_file(file_path, genre):
    try:
        features = extract_features(file_path)
        return (features, genre)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_gtzan_dataset(dataset_path):
    genres = os.listdir(dataset_path)
    data = []
    labels = []

    with multiprocessing.Pool() as pool:
        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            files = [os.path.join(genre_path, file) for file in os.listdir(genre_path)]

            # Use partial to pass the genre as an additional argument to process_file
            results = pool.map(partial(process_file, genre=genre), files)

            # Filter out None values (in case of errors)
            results = [result for result in results if result is not None]

            # Unpack the results into data and labels lists
            data.extend([result[0] for result in results])
            labels.extend([result[1] for result in results])

    return data, labels


def predict_genre(model, scaler, file):
    features = extract_features(file)
    features_scaled = scaler.transform([features])
    genre_pred = model.predict(features_scaled)
    return genre_pred[0]


if __name__ == '__main__':
    start_time = time.time()
    # Load the GTZAN dataset
    dataset_path = './DataSet/genres_original/'
    data, labels = load_gtzan_dataset(dataset_path)

    # Preprocess the dataset and implement feature scaling
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the SVM model
    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    new_song = "./SongsToTest/The Beatles - I Want To Hold Your Hand.wav"
    predicted_genre = predict_genre(model, scaler, new_song)
    print("Predicted genre:", predicted_genre)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total execution time: {elapsed_time:.2f} seconds")
