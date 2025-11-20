import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_polynomial_regression(X_train, y_train, X_test, y_test, degree=2):
    """
    Trains and evaluates a polynomial regression model.
    """
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Polynomial Regression (degree={degree}) Evaluated.")

    # FIX: Return a tuple containing the dictionary AND the model object
    return {'model': f'Polynomial Regression (d={degree})', 'RMSE': rmse, 'R-squared': r2}, model