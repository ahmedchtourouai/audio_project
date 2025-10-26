# !/bin/bash
# setup.sh

# Variables
REPO_NAME="audio_project"
DAGSHUB_USER="ahmedchtourou.ai"
DAGSHUB_TOKEN="4bf53230315d72afc16dd001b2188ae75c346f48"  # replace with your token
DATA_DIR="data/Tech Test"
PROCESSED_DIR="data/Tech Test/processed"

# 1️⃣ Initialize git
git init
git add .
git commit -m "Initial commit"

# 2️⃣ Initialize DVC
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"

# 3️⃣ Add raw dataset
dvc add "$DATA_DIR"
git add "$DATA_DIR.dvc"
git commit -m "Add raw audio dataset"

# 4️⃣ Configure DVC remote (DagsHub)
dvc remote add origin https://dagshub.com/$DAGSHUB_USER/$REPO_NAME.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $DAGSHUB_USER
dvc remote modify origin --local password $DAGSHUB_TOKEN

# 5️⃣ Run preprocessing script
python src/data_preprocessing.py

# 6️⃣ Track processed data
dvc add "$PROCESSED_DIR"
git add "$PROCESSED_DIR.dvc"
git commit -m "Add processed data"

# 7️⃣ Push datasets to DagsHub
dvc push -r origin

echo "✅ Setup complete! Datasets tracked and pushed to DagsHub"
