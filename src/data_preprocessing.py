# src/data_preprocessing.py
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

def extract_segments(audio_path, label_path, sr=16000, segment_dur=2.0, n_mels=128):
    """
    Extract mel spectrogram segments from an audio file based on its labels.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    samples, labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            segment = y[int(start * sr):int(end * sr)]
            target_len = int(segment_dur * sr)
            if len(segment) < target_len:
                segment = np.pad(segment, (0, target_len - len(segment)))
            else:
                segment = segment[:target_len]
            mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            samples.append(mel_db)
            labels.append(label)
    return np.array(samples), np.array(labels)

def load_data(audio_files, label_files, data_dir, main_classes=None):
    """
    Loads and processes audio data, filters based on specified classes.
    """
    X, y = [], []
    for wav, txt in zip(audio_files, label_files):
        mel, labels = extract_segments(data_dir / wav, data_dir / txt)
        X.append(mel)
        y.append(labels)

    X = np.concatenate(X)
    y = np.concatenate(y)
    print("✅ Original data loaded. Shape:", X.shape, "Labels:", np.unique(y))

    if main_classes:
        mask = np.isin(y, main_classes)
        X = X[mask]
        y = y[mask]
        print("✅ Filtered data. Shape:", X.shape, "Labels:", np.unique(y))

    return X, y

def preprocess_data(X, y):
    """
    Prepares data for CNN: reshapes X and encodes y labels.
    """
    X = X[..., np.newaxis]  # Adding channel dimension for CNN
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le
