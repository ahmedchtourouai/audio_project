

from tensorflow.keras import layers, models, optimizers
import optuna
import mlflow
import mlflow.keras
import os

# -------------------------------
# MODEL CREATION
# -------------------------------
def create_cnn_model(input_shape, n_classes, params=None, optimizer_name='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    if params is None:
        params = {"conv1_filters": 32, "conv2_filters": 64, "dense_units": 64, "learning_rate": 0.001}

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(params["conv1_filters"], (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(params["conv2_filters"], (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(params["dense_units"], activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    if optimizer_name.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=params["learning_rate"])
    elif optimizer_name.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=params["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    return model

# -------------------------------
# MODEL TRAINING WITH OPTUNA + DAGSHUB
# -------------------------------
def train_model(model, X_train, y_train, X_test, y_test, epochs=1, batch_size=8,
                use_optuna=False, n_trials=10, optimizer_name='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    """
    Trains the CNN model and logs parameters & metrics with MLflow (DagsHub compatible).
    """

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
            "loss": loss,
            "metrics": metrics
        })

        if use_optuna:
            def objective(trial):
                params = {
                    "conv1_filters": trial.suggest_categorical("conv1_filters", [16, 32, 64]),
                    "conv2_filters": trial.suggest_categorical("conv2_filters", [32, 64, 128]),
                    "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                }
                model_trial = create_cnn_model(X_train.shape[1:], y_train.max() + 1, params, optimizer_name, loss, metrics)
                history = model_trial.fit(X_train, y_train, validation_data=(X_test, y_test),
                                          epochs=epochs, batch_size=batch_size, verbose=0)
                mlflow.log_metrics({f"val_accuracy_trial_{trial.number}": history.history['val_accuracy'][-1]})
                return history.history['val_accuracy'][-1]

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            print("Best hyperparameters:", study.best_params)

            best_model = create_cnn_model(X_train.shape[1:], y_train.max() + 1, study.best_params, optimizer_name, loss, metrics)
            history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                     epochs=epochs, batch_size=batch_size, verbose=1)

            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"val_accuracy": history.history['val_accuracy'][-1],
                                "train_accuracy": history.history['accuracy'][-1]})
            return best_model, history

        else:
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                epochs=epochs, batch_size=batch_size, verbose=1)
            mlflow.keras.log_model(model, "cnn_model")
            mlflow.log_metrics({"val_accuracy": history.history['val_accuracy'][-1],
                                "train_accuracy": history.history['accuracy'][-1]})
            return model, history
