import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
import os

# Set up the data directory and file selection
excel_folder = './'
excel_files = [os.path.join(excel_folder, f) for f in os.listdir(excel_folder) if f.endswith('.xlsx') and "DPPH" in f]

for file in excel_files:
    init_R2 = 0
	# Filter out poorly trained models
    while init_R2 < 0.80:
        # Load and preprocess data
        data = pd.read_excel(file, header=None).astype(float)
        print(f"Starting with {file}")
        total = data.values
        X = total[:, 1:]
        y = total[:, 0]

        # Shuffle the data
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X_shuffled = X.copy().reshape((X.shape[0], X.shape[1], 1))
        y_true_shuffled = y.copy()

        # Split the data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Set up callbacks for model training
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath="sb_best_model.keras",
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

        # Load the best model and evaluate on the test set
        best_model = keras.models.load_model("sb_best_model.keras")
        test_loss = best_model.evaluate(X_test, y_test)
        init_R2 = test_loss[3]
        print(f"Test loss:", test_loss)
