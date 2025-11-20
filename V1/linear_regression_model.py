import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_linear_regression(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("Linear Regression Evaluated.")
    
    # FIX: Return a tuple containing the dictionary AND the model object
    return {'model': 'Linear Regression', 'RMSE': rmse, 'R-squared': r2}, model