import numpy as np
import pandas as pd
from itertools import product
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

def to_multidimensional_array(df : pd.DataFrame) -> np.array:
    """
    Convert a DataFrame to a multidimensional array.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert. 

    Returns:
    np.array: The multidimensional array.
    """
    # Sort the DataFrame by the index, so that the index is in the correct order once converted to a multidimensional array.
    df_sorted = df.sort_index()

    return df_sorted.values.reshape(
        *(df_sorted.index.levels[i].size for i in range(df_sorted.index.nlevels)),
        -1
        )

def enumerated_product(*args):
    """Enumerate the product of multiple iterables."""
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))