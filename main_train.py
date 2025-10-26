# ======================================================
# MLOps Audio Classification Proof of Concept
# Author: Ahmed Chtourou
# ======================================================

from src.data_preprocessing import load_data, preprocess_data
from src.model import create_cnn_model, train_model
from src.evaluation import evaluate_model
from src.inference import infer
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import mlflow
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import dagshub

# -------------------------------
# DAGSHUB CONFIGURATION
# -------------------------------

dagshub.init(repo_owner='ahmedchtourou.ai', repo_name='audio_project', mlflow=True)

DAGSHUB_TOKEN = "dd584fc53103f8d59effbcd85bd9b1e837fdf77f"  # optional if using token
if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_TOKEN"] = DAGSHUB_TOKEN

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Train and evaluate CNN for audio classification")
parser.add_argument("--data_dir", type=str, default="data/Tech Test", help="Path to folder with audio and label files")
parser.add_argument("--output_dir", type=str, default="output", help="Folder to save model, metrics, plots")
parser.add_argument("--use_optuna", type=bool, default=False, help="Whether to use Optuna for hyperparameter tuning")
parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer name: adam, sgd")
parser.add_argument("--loss", type=str, default="sparse_categorical_crossentropy", help="Loss function")
parser.add_argument("--metrics", nargs="+", default=["accuracy"], help="List of metrics")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--use_mlflow", type=bool, default=True, help="Enable MLflow tracking")

args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FILES = ["AS_1.wav", "23M74M.wav"]
LABEL_FILES = ["AS_1.txt", "23M74M.txt"]

def main():
    # Load data
    X, y = load_data(AUDIO_FILES, LABEL_FILES, DATA_DIR, main_classes=['b', 'mb', 'h'])

    # Preprocess data
    X, y_encoded, le = preprocess_data(X, y)
# Save preprocessed data for DVC tracking
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y_encoded)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # MLflow experiment
    if args.use_mlflow:
        mlflow.set_experiment("audio_classification_experiment")

    # Create model
    model = create_cnn_model(
        input_shape=(X.shape[1], X.shape[2], 1),
        n_classes=len(le.classes_),
        optimizer_name=args.optimizer,
        loss=args.loss,
        metrics=args.metrics
    )

    # Train model
    model, history = train_model(
        model,
        X_train, y_train,
        X_test, y_test,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        optimizer_name=args.optimizer,
        loss=args.loss,
        metrics=args.metrics,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Save model locally
    model_path = OUTPUT_DIR / "cnn_audio_classifier.h5"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # Evaluate model
    evaluate_model(model, X_test, y_test, OUTPUT_DIR)

    # Inference example
    predicted_class = infer(model, DATA_DIR / AUDIO_FILES[0], DATA_DIR / LABEL_FILES[0])
    print(f"Predicted class: {le.inverse_transform(predicted_class)}")


if __name__ == "__main__":
    main()
