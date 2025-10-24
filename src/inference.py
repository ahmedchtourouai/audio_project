from src.data_preprocessing import extract_segments
import numpy as np
# src/inference.py
def infer(model, audio_path, label_path, sr=16000):
    """
    Performs inference on a single audio sample.
    """
    mel, _ = extract_segments(audio_path, label_path)
    mel = mel[..., np.newaxis]  # Add channel dimension for CNN
    prediction = model.predict(mel)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class
