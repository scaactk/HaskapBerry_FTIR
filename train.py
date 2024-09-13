import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
import keras
import os

# Set up the data directory and file selection
excel_folder = './'
excel_files = [os.path.join(excel_folder, f) for f in os.listdir(excel_folder) if f.endswith('.xlsx') and "ORAC" in f]

for file in excel_files:
    # Load and preprocess data
    data = pd.read_excel(file, header=None).astype(float)
    total = data.values
    X = total[:, 1:]
    y = total[:, 0]

    # Shuffle the data
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    y_true_shuffled = y.copy()

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    all_predictions = np.zeros_like(y)
    fold_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"Training on fold {fold}")
        print("Validation indices:", val_index)

        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Reshape input data for CNN
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        # Set up callbacks for model training
        check_point_name = f"{file.replace('.xlsx', '')}_fold_{fold}"
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f"{check_point_name}_best_model.keras",
                save_best_only=True,
                monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.00001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=100, verbose=2),
        ]

		input = keras.layers.Input(shape=[X_train.shape[1]])
		x = keras.layers.Reshape((X_train.shape[1], 1))(input)
		x = keras.layers.Conv1D(128, 100, activation='relu')(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.ReLU()(x)

		x = keras.layers.Conv1D(64, 100, activation='relu')(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.ReLU()(x)

		x = keras.layers.Conv1D(64, 100, activation='relu')(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.ReLU()(x)

		x = keras.layers.Conv1D(64, 100, activation='relu')(x)
		x = keras.layers.GlobalAvgPool1D()(x)
		x = keras.layers.Flatten()(x)
		x = keras.layers.Dense(10, activation='relu')(x)
		output = keras.layers.Dense(1)(x)

        model = keras.Model(input_layer, output)

        # Compile the model
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
            metrics=[keras.metrics.MeanAbsoluteError(),
                     keras.metrics.RootMeanSquaredError(),
                     keras.metrics.R2Score()],
        )

        # Train the model
        model.fit(
            X_train,
            y_train,
            batch_size=4,
            epochs=1000,
            callbacks=callbacks,
            validation_data=(X_val, y_val),
            verbose=2,
        )

        # Load the best model and evaluate on the validation set
        best_model = keras.models.load_model(f"{check_point_name}_best_model.keras")
        val_loss = best_model.evaluate(X_val, y_val)
        print(f"Fold {fold} validation loss:", val_loss)
        fold_scores.append(val_loss)

        # Predict on the validation set
        all_predictions[val_index] = best_model.predict(X_val).flatten()
        print("Validation results:", all_predictions[val_index])
        print("Validation true values:", y_val)

    # Print average scores across all folds
    print("Average scores across all folds:", np.mean(fold_scores, axis=0))

    # Save predictions
    data_to_save = np.column_stack((y_true_shuffled, all_predictions))
    np.savetxt(f"{file.replace('.xlsx', '')}_predictions.csv", data_to_save, delimiter=",", header="y_true,predictions", comments='')

    print(f"Predictions for {file} have been saved.")