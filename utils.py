import numpy as np
import pandas as pd

def geometric_series(b, n):
    # Create a range of exponents from 0 to n-1
    exponents = np.arange(n)
    # Compute b to the power of each exponent
    series = np.power(b, exponents)
    return series

def set_column_index_name(df, name):
    """
    Set the name of the columns index of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame.
    name (str): The name to set for the columns index.
    
    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    df_copy = df.copy()
    df_copy.columns.name = name
    return df_copy

