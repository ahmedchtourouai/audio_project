# # ======================================================
# # MLOps Audio Classification Proof of Concept (Filtered to main classes)
# # Author: Ahmed Chtourou
# # ======================================================

# # --- 1. Imports ---
# import os
# import argparse
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import json
# from pathlib import Path

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress info logs

# # --- 2. Command-line arguments ---
# parser = argparse.ArgumentParser(description="Train CNN for audio classification")
# parser.add_argument("--data_dir", type=str, default="data/Tech Test", help="Path to folder with audio and label files")
# parser.add_argument("--output_dir", type=str, default="output", help="Folder to save model, metrics, plots")
# args = parser.parse_args()

# DATA_DIR = Path(args.data_dir)
# OUTPUT_DIR = Path(args.output_dir)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # --- 3. Files ---
# AUDIO_FILES = ["AS_1.wav", "23M74M.wav"]
# LABEL_FILES = ["AS_1.txt", "23M74M.txt"]

# # --- 4. Parameters ---
# SR = 16000         # Sampling rate
# N_MELS = 128       # Number of Mel bands
# SEGMENT_DUR = 2.0  # Seconds per training sample

# # --- 5. Function: extract mel-spectrograms ---
# def extract_segments(audio_path, label_path):
#     y, sr = librosa.load(audio_path, sr=SR)
#     samples, labels = [], []
#     with open(label_path, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 3:
#                 continue
#             start, end, label = float(parts[0]), float(parts[1]), parts[2]
#             segment = y[int(start * sr):int(end * sr)]
#             target_len = int(SEGMENT_DUR * sr)
#             if len(segment) < target_len:
#                 segment = np.pad(segment, (0, target_len - len(segment)))
#             else:
#                 segment = segment[:target_len]
#             mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS)
#             mel_db = librosa.power_to_db(mel, ref=np.max)
#             samples.append(mel_db)
#             labels.append(label)
#     return np.array(samples), np.array(labels)

# # --- 6. Load dataset ---
# X, y = [], []
# for wav, txt in zip(AUDIO_FILES, LABEL_FILES):
#     mel, labels = extract_segments(DATA_DIR / wav, DATA_DIR / txt)
#     X.append(mel)
#     y.append(labels)

# X = np.concatenate(X)
# y = np.concatenate(y)
# print("✅ Original data loaded. Shape:", X.shape, "Labels:", np.unique(y))

# # --- 7. Filter main classes ---
# MAIN_CLASSES = ['b', 'mb', 'h']
# mask = np.isin(y, MAIN_CLASSES)
# X = X[mask]
# y = y[mask]
# print("✅ Filtered data. Shape:", X.shape, "Labels:", np.unique(y))

# # --- 8. Prepare for CNN ---
# X = X[..., np.newaxis]
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
# )

# # --- 9. Define CNN ---
# model = models.Sequential([
#     layers.Input(shape=(N_MELS, X.shape[2], 1)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(len(le.classes_), activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # --- 10. Train Model ---
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=3,
#     batch_size=8,
#     verbose=1
# )

# # --- 11. Evaluation ---
# y_pred = np.argmax(model.predict(X_test), axis=1)
# report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

# # Save metrics
# metrics_path = OUTPUT_DIR / "metrics.json"
# with open(metrics_path, "w") as f:
#     json.dump(report, f, indent=4)
# print(f"✅ Metrics saved to {metrics_path}")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5,4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=le.classes_, yticklabels=le.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
# plt.close()
# print(f"✅ Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")

# # --- 12. Save model ---
# model_path = OUTPUT_DIR / "cnn_audio_classifier.h5"
# model.save(model_path)
# print(f"✅ Model saved to {model_path}")



# src/main.py
from src.data_preprocessing import load_data, preprocess_data
from src.model import create_cnn_model, train_model
from src.evaluation import evaluate_model
from src.inference import infer
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # suppress info logs

# Command-line arguments
parser = argparse.ArgumentParser(description="Train and evaluate CNN for audio classification")
parser.add_argument("--data_dir", type=str, default="data/Tech Test", help="Path to folder with audio and label files")
parser.add_argument("--output_dir", type=str, default="output", help="Folder to save model, metrics, plots")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio files and label files
AUDIO_FILES = ["AS_1.wav", "23M74M.wav"]
LABEL_FILES = ["AS_1.txt", "23M74M.txt"]

def main():
    # Load data
    X, y = load_data(AUDIO_FILES, LABEL_FILES, DATA_DIR, main_classes=['b', 'mb', 'h'])

    # Preprocess data
    X, y_encoded, le = preprocess_data(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # Create and train the model
    model = create_cnn_model((X.shape[1], X.shape[2], 1), len(le.classes_))
    train_model(model, X_train, y_train, X_test, y_test)

    # Save the model
    model_path = OUTPUT_DIR / "cnn_audio_classifier.h5"
    
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # Evaluate the model
    evaluate_model(model, X_test, y_test, OUTPUT_DIR)

    # Inference on a new sample
    predicted_class = infer(model, DATA_DIR / AUDIO_FILES[0], DATA_DIR / LABEL_FILES[0])
    print(f"Predicted class: {le.inverse_transform(predicted_class)}")

if __name__ == "__main__":
    main()
