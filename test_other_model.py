import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Set up the data directory and file selection
excel_folder = './'
excel_files = [os.path.join(excel_folder, f) for f in os.listdir(excel_folder) if f.endswith('.xlsx')]

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model, returning performance metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

for file in excel_files:
    print(f"Processing {file}")
    # Load and preprocess data
    data = pd.read_excel(file, header=None).astype(float)
    total = data.values
    X = total[:, 1:]
    y = total[:, 0]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf')
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        mse_sum = mae_sum = r2_sum = 0
        t = 5  # Number of iterations for averaging results
        for _ in range(t):
            mse, mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
            mse_sum += mse
            mae_sum += mae
            r2_sum += r2
        results[name] = {'MSE': mse_sum/t, 'MAE': mae_sum/t, 'R2': r2_sum/t}

    # Print results
    print(f"\nResults for {file}:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  R2: {metrics['R2']:.4f}")
        print()
