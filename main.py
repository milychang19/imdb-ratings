import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# Import the functions from our modules
from sampling import create_train_test_split
from linear_regression_model import train_and_evaluate_linear_regression
from polynomial_regression_model import train_and_evaluate_polynomial_regression
from lightgbm_model import train_and_evaluate_lightgbm

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Saves a scatter plot of actual vs. predicted values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual IMDb Score')
    plt.ylabel('Predicted IMDb Score')
    plt.title(f'Actual vs. Predicted Scores - {model_name}')
    plt.savefig(f'charts/{model_name.replace(" ", "_").lower()}_actual_vs_predicted.png')
    plt.close()

def plot_residuals(y_test, y_pred, model_name):
    """Saves a plot of residuals vs. predicted values."""
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, 
                  scatter_kws={'alpha': 0.5}, 
                  line_kws={'color': 'red', 'lw': 2})
    plt.xlabel('Predicted IMDb Score')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residual Plot - {model_name}')
    plt.savefig(f'charts/{model_name.replace(" ", "_").lower()}_residuals.png')
    plt.close()

def plot_shap_summary(model, X_test, feature_names):
    """Saves a SHAP summary plot to explain feature importance."""
    print("Generating SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('charts/lgbm_shap_summary.png')
    plt.close()

def prepare_data(movies_path, directors_path, actors_path):
    """Loads, merges, and prepares the movie data for modeling."""
    movies_df = pd.read_csv(movies_path)
    directors_df = pd.read_csv(directors_path)
    actors_df = pd.read_csv(actors_path)
    data = pd.merge(movies_df, directors_df, on='director_name', how='left')
    data = pd.merge(data, actors_df.add_suffix('_1'), left_on='actor_1_name', right_on='actor_name_1', how='left')
    data = pd.merge(data, actors_df.add_suffix('_2'), left_on='actor_2_name', right_on='actor_name_2', how='left')
    data = pd.merge(data, actors_df.add_suffix('_3'), left_on='actor_3_name', right_on='actor_name_3', how='left')
    
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    features_to_keep = [col for col in numeric_cols if col not in ['title_year']]
    
    # --- FIX IS HERE ---
    # Add .copy() to explicitly create a new DataFrame and avoid the warning.
    final_df = data[features_to_keep].copy()
    
    for col in final_df.columns:
        if final_df[col].isnull().any():
            final_df[col] = final_df[col].fillna(final_df[col].mean())
    return final_df

if __name__ == "__main__":
    # --- Setup ---
    if not os.path.exists('charts'):
        os.makedirs('charts')

    MOVIES_CSV = 'imdb_movies_final.csv'
    DIRECTORS_CSV = 'director_summary.csv'
    ACTORS_CSV = 'actors_summary.csv'

    model_data = prepare_data(MOVIES_CSV, DIRECTORS_CSV, ACTORS_CSV)
    TARGET = 'imdb_score'

    X_train, X_test, y_train, y_test = create_train_test_split(model_data, TARGET)
    
    print("\n--- Starting Model Training, Evaluation, and Visualization ---")
    
    # --- Linear Regression ---
    lr_results, lr_model = train_and_evaluate_linear_regression(X_train, y_train, X_test, y_test)
    y_pred_lr = lr_model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_lr, 'Linear Regression')
    plot_residuals(y_test, y_pred_lr, 'Linear Regression')
    print("Linear Regression charts saved.")

    # --- Polynomial Regression ---
    poly_results, poly_model = train_and_evaluate_polynomial_regression(X_train, y_train, X_test, y_test, degree=2)
    y_pred_poly = poly_model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_poly, 'Polynomial Regression (d=2)')
    plot_residuals(y_test, y_pred_poly, 'Polynomial Regression (d=2)')
    print("Polynomial Regression charts saved.")

    # --- LightGBM ---
    lgbm_results, lgbm_model = train_and_evaluate_lightgbm(X_train.values, y_train, X_test.values, y_test)
    y_pred_lgbm = lgbm_model.predict(X_test.values)
    plot_actual_vs_predicted(y_test, y_pred_lgbm, 'LightGBM')
    plot_residuals(y_test, y_pred_lgbm, 'LightGBM')
    plot_shap_summary(lgbm_model, X_test.values, X_train.columns)
    print("LightGBM charts saved.")

    # --- Display Final Results ---
    results_df = pd.DataFrame([lr_results, poly_results, lgbm_results])
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string(index=False))