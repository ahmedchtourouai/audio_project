# # ======================================================
# # MLOps Audio Classification Proof of Concept (Filtered to main classes)
# # Author: Ahmed Chtourou
# # ======================================================
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
