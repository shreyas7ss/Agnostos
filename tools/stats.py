"""
Stats Tool - Automated plotting and statistical profiling tools
Generates visualizations and computes statistics for data analysis
"""

#the data cleaning will be done by the scientist agent

import pandas as pd
import numpy as np

def tabular_profiler(file_path: str) -> dict:
    """
    Calculates deep stats for CSV/Parquet files.
    """
    df = pd.read_csv(file_path)
    
    # Basic Metadata
    stats = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }
    
    # Statistical Summary
    # We use .to_dict() on describe to help the Scientist understand distributions
    stats["numeric_summary"] = df.describe().to_dict()
    
    # Correlation (Crucial for feature selection)
    # We only take numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats["correlation_matrix"] = numeric_df.corr().to_dict()
        
    # Categorical Analysis (Crucial for Encoding decisions)
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    stats["categorical_cardinality"] = {col: df[col].nunique() for col in cat_columns}
    
    return stats