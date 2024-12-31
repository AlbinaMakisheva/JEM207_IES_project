from sklearn.linear_model import LinearRegression
import numpy as np

def perform_regression_analysis(data, independent_var='new_vaccinations_smoothed_per_million', dependent_var='daily_return'):
    if independent_var not in data.columns or dependent_var not in data.columns:
        raise KeyError(f"Columns '{independent_var}' or '{dependent_var}' not found in the dataset.")
    
    # Drop empty values
    regression_data = data[[independent_var, dependent_var]].dropna()

    X = regression_data[independent_var].values.reshape(-1, 1)
    y = regression_data[dependent_var].values

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    print(f"Regression Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R^2 Score: {model.score(X, y):.4f}")
    return model
