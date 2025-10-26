# Audio Classification Project

```bash
1️⃣ Create a Virtual Environment
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




🖥️ 8️⃣ CLI Training

The training script main.py supports incremental training scenarios.
You can choose different modes for how the model is trained:

Arguments:
Argument	Type	Default	Description
--data_dir	str	data/Tech Test	Folder with audio and label files
--output_dir	str	output	Folder to save models, metrics, plots
--new_data_files	list[str]	[]	New audio files for prediction/retraining
--train_mode	str	old_only	Training scenario: old_only, semi_label, expert_label
--use_optuna	bool	False	Enable Optuna hyperparameter tuning
--n_trials	int	10	Number of Optuna trials
--optimizer	str	adam	Optimizer name
--loss	str	sparse_categorical_crossentropy	Loss function
--metrics	list[str]	['accuracy']	List of metrics
--batch_size	int	8	Batch size for training
--epochs	int	10	Number of training epochs
--use_mlflow	bool	True	Enable MLflow tracking
Example Usages:
1️⃣ Train only on initial data
python main.py --train_mode old_only

2️⃣ Train with semi-labeled new data
python main.py --train_mode semi_label --new_data_files 23M74M.wav

3️⃣ Train with expert-labeled new data
python main.py --train_mode expert_label --new_data_files 23M74M.wav


✅ In all cases, models are saved under output/YYYY-MM-DD/ with separate names for each scenario.

📂 Folder Structure
audio_project/
│
├─ data/                  # Input audio and label files
│   ├─ Tech Test/
│   │   ├─ AS_1.wav
│   │   ├─ AS_1.txt
│   │   └─ 23M74M.wav
│
├─ output/                # Models and metrics are saved here
│   └─ YYYY-MM-DD/
│       ├─ cnn_model_initial.h5
│       ├─ cnn_model_semi_label.h5
│       └─ cnn_model_expert_label.h5
│
├─ src/
│   ├─ data_preprocessing.py
│   ├─ model.py
│   ├─ evaluation.py
│   └─ inference.py
│
├─ main.py                # Main training script
├─ requirements.txt
└─ README.md

📈 Incremental Training & Semi-Automated Labeling Workflow

Train the initial model with existing labeled data (old_only).

When new user data arrives:

Predict targets with the old model.

Optionally retrain the model with semi-labeled data (semi_label).

Once the expert annotates new data:

Retrain the model with correct labels (expert_label).

Use MLflow to track experiments and metrics for each scenario.

Deploy the best performing model to production.

🔗 (MLflow& dagshub) & Experiment Tracking

All experiments are logged automatically to MLflow.

Track metrics, losses, and models.


⚡ Notes

Models are saved in a date-stamped folder under output/.

CLI arguments allow flexible retraining without editing the script.

Docker images are provided for training and inference.
