import pandas as pd
import numpy as np

def apply_low_resolution(df, features, bins=5):
    """
    Simulates low-precision instruments by compressing continuous decimals into discrete chunks.
    Tests if the scientific pattern survives resolution degradation.
    """
    perturbed_df = df.copy()
    for feature in features:
        # pd.qcut divides the data into statistical buckets (quantiles).
        # This destroys hyper-specific decimal precision.
        perturbed_df[feature] = pd.qcut(perturbed_df[feature], q=bins, labels=False, duplicates='drop')
    return perturbed_df

def isolate_extremes(df, target, percentile=0.10):
    """
    Removes all normal data. Tests if the hypothesis holds true during 
    extreme anomalies (the top 10% and bottom 10% of the target variable).
    """
    # Calculate the mathematical boundaries for the extremes
    lower_bound = df[target].quantile(percentile)
    upper_bound = df[target].quantile(1 - percentile)
    
    # Filter the dataset to only include rows that fall outside the normal boundaries
    extreme_df = df[(df[target] <= lower_bound) | (df[target] >= upper_bound)]
    return extreme_df

def apply_domain_shift(df, primary_feature):
    """
    Splits the universe in half based on the AI's cognitive anchor.
    Tests if the hypothesis collapses when the main driving force is constrained.
    """
    median_val = df[primary_feature].median()
    
    # We force the model to operate strictly in the 'High State' of the environment
    shifted_df = df[df[primary_feature] > median_val]
    return shifted_df