# src/api.py
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
from src.data_preprocessing import extract_segments
from pathlib import Path
import shutil
import tempfile
import numpy as np
import traceback
import librosa

app = FastAPI()

MODEL_PATH = Path("output/cnn_audio_classifier.h5")
model = load_model(MODEL_PATH)

# Define your main classes
MAIN_CLASSES = ['b', 'mb', 'h']

@app.get("/health/")
async def health_check():
    """
    Simple health check endpoint.
    Returns the API status and model readiness.
    """
    try:
        model_ready = model is not None
        return {"status": "ok", "model_loaded": model_ready}
    except Exception as e:
        return {"status": "error", "details": str(e)}

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        # Step 1: Save uploaded WAV to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            shutil.copyfileobj(file.file, tmp_wav)
            tmp_wav_path = Path(tmp_wav.name)

        # Step 2: Create a dummy label file with one segment covering the full audio
        duration = librosa.get_duration(filename=tmp_wav_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
            tmp_txt.write(f"0 {duration} {MAIN_CLASSES[0]}\n".encode())
            tmp_txt_path = Path(tmp_txt.name)

        # Step 3: Extract segments (Mel-spectrogram)
        mel_segments, _ = extract_segments(tmp_wav_path, tmp_txt_path)

        # Filter only main classes (optional, if your model was trained on them)
        # Here we just assume dummy label matches a main class
        X = mel_segments[..., np.newaxis]  # add channel dimension
        prediction = model.predict(X)
        predicted_indices = np.argmax(prediction, axis=1)
        predicted_classes = [MAIN_CLASSES[i] if i < len(MAIN_CLASSES) else "unknown"
                                    for i in predicted_indices]

        # Step 4: Clean up temp files
        tmp_wav_path.unlink()
        tmp_txt_path.unlink()

        # Step 5: Return predictions
        return {"predicted_class": predicted_classes}

    except Exception as e:
        print("❌ Error during prediction:", e)
        traceback.print_exc()
        return {"error": str(e)}
