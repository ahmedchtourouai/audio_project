1️⃣ Create the virtual environment
python -m venv mlops_env
2️⃣ Activate the virtual environment in PowerShell
.\mlops_env\Scripts\Activate.ps1
3️⃣ Install dependencies
pip install --upgrade pip or  python.exe -m pip install --upgrade pip
pip install -r requirements.txt
4️⃣ Deactivate when done
deactivate

uvicorn api.app:app --reload

docker build -f src/dockerfile.train -t audio_train:latest .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  audio_train:latest


docker build -f api/Dockerfile.api -t audio_infer:latest .
docker run --rm -p 8000:8000 audio_infer:latest
