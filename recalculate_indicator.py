import os

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Get all prediction files
results_files = [os.path.join('./', f) for f in os.listdir('./') if '_predictions' in f and 'Poly' in f]
print(results_files)

for file in results_files:
	y_pred = pd.read_csv(file, header=None)[0]
	y_true = pd.read_excel(file.replace("_predictions.csv", "") + ".xlsx", header=None)[0]
	print(y_pred)
	print(y_true)

	# Calculate evaluation metrics
	mse = mean_squared_error(y_true, y_pred)
	mae = mean_absolute_error(y_true, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_true, y_pred)

	print(file)
	print(f"Mean Squared Error: {mse}")
	print(f"Mean Absolute Error: {mae}")
	print(f"Root Mean Squared Error: {rmse}")
	print(f"R2 Score: {r2}")
