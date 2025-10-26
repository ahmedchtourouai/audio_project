import requests
from pathlib import Path
import wandb
print(wandb.__version__)

url = "http://127.0.0.1:8000/predict/"
file_path = Path("data/Tech Test/AS_1.wav")

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response:", response.json())
