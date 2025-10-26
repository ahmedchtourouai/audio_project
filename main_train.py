# ======================================================
# MLOps Audio Classification Proof of Concept
# Incremental Training + Semi-Automated Labeling
# Author: Ahmed Chtourou
# ======================================================

from src.data_preprocessing import load_data, preprocess_data
from src.model import create_cnn_model, train_model
from src.evaluation import evaluate_model
from src.inference import infer
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import os
import mlflow
import numpy as np
import dagshub
from datetime import datetime
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# -------------------------------
# DAGSHUB CONFIGURATION
# -------------------------------
dagshub.init(repo_owner='ahmedchtourou.ai', repo_name='audio_project', mlflow=True)

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Incremental Audio Classification Training")
parser.add_argument("--data_dir", type=str, default="data/Tech Test", help="Folder with audio and label files")
parser.add_argument("--output_dir", type=str, default="output", help="Folder to save model, metrics, plots")
parser.add_argument("--new_data_files", nargs="+", default=[], help="New audio files for prediction/retraining")
parser.add_argument("--train_mode", type=str, choices=["old_only", "semi_label", "expert_label"],
                    default="old_only", help="Training scenario: old_only / semi_label / expert_label")
parser.add_argument("--use_optuna", type=bool, default=False, help="Enable Optuna hyperparameter tuning")
parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--loss", type=str, default="sparse_categorical_crossentropy")
parser.add_argument("--metrics", nargs="+", default=["accuracy"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--use_mlflow", type=bool, default=True)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(args.output_dir) / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Initial Data
# -------------------------
INITIAL_AUDIO = ["AS_1.wav"]
INITIAL_LABELS = ["AS_1.txt"]
MAIN_CLASSES = ['b', 'mb', 'h']

def main():
    logging.info("===== Starting Audio Classification Incremental Training =====")
    
    # -------------------------
    # Step 1: Load and preprocess initial data
    # -------------------------
    logging.info("Loading initial data...")
    X_old, y_old = load_data(INITIAL_AUDIO, INITIAL_LABELS, DATA_DIR, main_classes=MAIN_CLASSES)
    X_old, y_old_encoded, le = preprocess_data(X_old, y_old)
    logging.info(f"Loaded {len(X_old)} audio samples from old dataset.")

    if args.use_mlflow:
        mlflow.set_experiment("audio_classification_incremental")

    # -------------------------
    # Train only on old data
    # -------------------------
    if args.train_mode == "old_only":
        logging.info("Training scenario: old_only")
        X_train, X_test, y_train, y_test = train_test_split(X_old, y_old_encoded, test_size=0.2,
                                                            stratify=y_old_encoded, random_state=42)
        model = create_cnn_model(input_shape=(X_old.shape[1], X_old.shape[2], 1),
                                 n_classes=len(le.classes_))
        model, _ = train_model(model, X_train, y_train, X_test, y_test,
                                use_optuna=args.use_optuna, n_trials=args.n_trials,
                                optimizer_name=args.optimizer, loss=args.loss, metrics=args.metrics,
                                batch_size=args.batch_size, epochs=args.epochs)
        model_path = OUTPUT_DIR / "cnn_model_initial.h5"
        model.save(model_path)
        logging.info(f"Model trained on old data only: {model_path}")
        evaluate_model(model, X_test, y_test, OUTPUT_DIR)
        logging.info("Evaluation completed for old_only model.")

    # -------------------------
    # Semi-labeled new data scenario
    # -------------------------
    elif args.train_mode == "semi_label" and args.new_data_files:
        logging.info("Training scenario: semi_label")
        X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old_encoded, test_size=0.2,
                                                                            stratify=y_old_encoded, random_state=42)
        model_old = create_cnn_model(input_shape=(X_old.shape[1], X_old.shape[2], 1),
                                     n_classes=len(le.classes_))
        model_old, _ = train_model(model_old, X_train_old, y_train_old, X_test_old, y_test_old,
                                    use_optuna=args.use_optuna, n_trials=args.n_trials,
                                    optimizer_name=args.optimizer, loss=args.loss, metrics=args.metrics,
                                    batch_size=args.batch_size, epochs=args.epochs)
        logging.info("Old model trained for semi-label predictions.")

        # Predict new data
        X_new_list, y_pred_list = [], []
        for new_audio in args.new_data_files:
            new_label_file = new_audio.replace(".wav", ".txt")
            logging.info(f"Predicting pseudo-labels for new data: {new_audio}")
            X_new, _ = load_data([new_audio], [new_label_file], DATA_DIR, main_classes=MAIN_CLASSES)
            X_new, _, _ = preprocess_data(X_new, np.array(['dummy']*len(X_new)))
            y_pred = model_old.predict(X_new).argmax(axis=1)
            X_new_list.append(X_new)
            y_pred_list.append(y_pred)
            np.save(OUTPUT_DIR / f"pred_{new_audio}.npy", y_pred)
            logging.info(f"Predictions saved for {new_audio}")

        # Combine old + predicted new data
        X_combined = np.concatenate([X_old] + X_new_list)
        y_combined = np.concatenate([y_old_encoded] + y_pred_list)

        # Retrain on combined data
        model_semi = create_cnn_model(input_shape=(X_combined.shape[1], X_combined.shape[2], 1),
                                      n_classes=len(le.classes_))
        model_semi, _ = train_model(model_semi, X_combined, y_combined, X_combined, y_combined,
                                    batch_size=args.batch_size, epochs=args.epochs)
        model_path = OUTPUT_DIR / "cnn_model_semi_label.h5"
        model_semi.save(model_path)
        logging.info(f"Model retrained with semi-labeled new data: {model_path}")
        evaluate_model(model_semi, X_combined, y_combined, OUTPUT_DIR)
        logging.info("Evaluation completed for semi-labeled model.")

    # -------------------------
    # Expert-labeled new data scenario
    # -------------------------
    elif args.train_mode == "expert_label" and args.new_data_files:
        logging.info("Training scenario: expert_label")
        X_new_list, y_new_list = [], []
        for new_audio in args.new_data_files:
            new_label_file = new_audio.replace(".wav", ".txt")
            logging.info(f"Loading expert-labeled new data: {new_audio}")
            X_new, y_new = load_data([new_audio], [new_label_file], DATA_DIR, main_classes=MAIN_CLASSES)
            X_new, y_new_encoded, _ = preprocess_data(X_new, y_new)
            X_new_list.append(X_new)
            y_new_list.append(y_new_encoded)

        # Combine old + expert-labeled new data
        X_combined = np.concatenate([X_old] + X_new_list)
        y_combined = np.concatenate([y_old_encoded] + y_new_list)

        # Train final model
        model_final = create_cnn_model(input_shape=(X_combined.shape[1], X_combined.shape[2], 1),
                                       n_classes=len(le.classes_))
        model_final, _ = train_model(model_final, X_combined, y_combined, X_combined, y_combined,
                                     batch_size=args.batch_size, epochs=args.epochs)
        model_path = OUTPUT_DIR / "cnn_model_expert_label.h5"
        model_final.save(model_path)
        logging.info(f"Model trained with expert-labeled new data: {model_path}")
        evaluate_model(model_final, X_combined, y_combined, OUTPUT_DIR)
        logging.info("Evaluation completed for expert-labeled model.")

    else:
        logging.warning("No training executed. Check --train_mode and new data files.")

    logging.info("===== Training Process Completed =====")

if __name__ == "__main__":
    main()
