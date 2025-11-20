import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_and_evaluate_lightgbm(X_train, y_train, X_test, y_test):
    """
    Trains, evaluates, and plots the learning curve for a LightGBM model.
    """
    model = lgb.LGBMRegressor(random_state=42)
    
    # Train the model, evaluating on the test set to track learning
    model.fit(X_train, y_train.values.ravel(),
              eval_set=[(X_test, y_test.values.ravel())],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(period=10)])

    # --- Plotting Training Loss ---
    print("Generating LightGBM training loss plot...")
    plt.figure(figsize=(10, 6))
    lgb.plot_metric(model, metric='rmse', title='LightGBM Training and Validation Loss (RMSE)')
    plt.savefig('charts/lgbm_loss_curve.png')
    plt.close()
    
    # Make final predictions
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("LightGBM Evaluated.")
    # Return both the metrics dictionary and the trained model object
    return {'model': 'LightGBM', 'RMSE': rmse, 'R-squared': r2}, model