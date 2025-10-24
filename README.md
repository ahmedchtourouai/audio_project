# Audio Classification Project
1️⃣ Create a Virtual Environment

```bash
python -m venv mlops_env
2️⃣ Activate the Virtual Environment (PowerShell)

.\mlops_env\Scripts\Activate.ps1
3️⃣ Install Dependencies
Upgrade pip:


pip install --upgrade pip
# or
python.exe -m pip install --upgrade pip
Install required packages:


pip install -r requirements.txt
4️⃣ Deactivate Virtual Environment

deactivate
5️⃣ Run API Locally

uvicorn api.app:app --reload
6️⃣ Docker: Training
Build the Docker image:


docker build -f src/dockerfile.train -t audio_train:latest .
Run the container with mounted volumes:


docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  audio_train:latest
7️⃣ Docker: Inference/API
Build the Docker image:


docker build -f api/Dockerfile.api -t audio_infer:latest .
Run the container and expose port 8000:


docker run --rm -p 8000:8000 audio_infer:latest