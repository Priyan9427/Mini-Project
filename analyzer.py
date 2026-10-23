import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import shap
import statsmodels.api as sm

def calculate_metrics(df, features, target):
    """
    A Hybrid Dual-Engine Analyzer.
    Engine 1 (Stats): Calculates P-Values for scientific probability.
    Engine 2 (AI): Uses XGBoost & SHAP for non-linear pattern recognition.
    """
    X = df[features]
    y = df[target]
    
    # ==========================================
    # ENGINE 1: TRADITIONAL STATISTICAL MATH
    # ==========================================
    # sm.add_constant adds a y-intercept to the math, which is required for accurate statistics
    X_stat = sm.add_constant(X)
    
    # sm.OLS (Ordinary Least Squares) is classical linear regression. 
    # .fit() calculates the strict statistical probabilities.
    stat_model = sm.OLS(y, X_stat).fit()
    
    # We extract the P-values for all inputs, drop the constant, and find the highest one.
    # In science, if any P-value is above 0.05, the pattern is considered mathematically invalid.
    max_p_value = stat_model.pvalues.drop('const', errors='ignore').max()

    # ==========================================
    # ENGINE 2: MACHINE LEARNING & AI SAFETY
    # ==========================================
    model = xgb.XGBRegressor(
        n_estimators=100, 
        tree_method='hist', 
        device='cuda',
        random_state=42
    )
    model.fit(X, y)
    
    r2 = model.score(X, y)
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    # --- SHAP Explainable AI ---
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)
    
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[0]
    else:
        shap_values = shap_values_raw
        
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    total_importance = np.sum(mean_shap_values)
    
    feature_importance = {}
    for i, feature in enumerate(features):
        if total_importance == 0:
            percentage = 0
        else:
            percentage = (mean_shap_values[i] / total_importance) * 100
        feature_importance[feature] = percentage
        
    # We now return BOTH statistical and AI metrics
    return {
        "r2": r2,
        "rmse": rmse,
        "p_value": max_p_value, # The new statistical metric
        "importance": feature_importance
    }