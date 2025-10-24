# src/evaluation.py
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluates the model and saves metrics and confusion matrix.
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"✅ Metrics saved to {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    print(f"✅ Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
