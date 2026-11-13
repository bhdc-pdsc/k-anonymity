from typing import List, Optional

import numpy as np
import pandas as pd


def convert_numerical_to_categorical(df: pd.DataFrame, column_name: str, bin_size: Optional[int] = None,
                                     categories_thresholds: Optional[List[float]] = None) -> pd.DataFrame:
    """
    input: dataframe, name of the numerical column to convert to categorical, number of bins for categorization or list of categories thresholds
    create categorical version based either on the number of bins or thresholds defined in a list
    output: copy of the dataframe with the new column <column_name>_categorical (the input dataframe is not modified)
    """
    if bin_size and categories_thresholds:
        raise ValueError("Both bin_size and categories_thresholds are provided, please provide only one.")
    if not bin_size and not categories_thresholds:
        raise ValueError("No bin_size or categories_thresholds provided, please provide one (but only one).")

    df = df.copy()
    categorical_name = column_name + "_categorical"
    if bin_size:
        df[categorical_name] = pd.cut(df[column_name], bins=bin_size)
    else:
        thresholds = sorted(categories_thresholds)
        col_min, col_max = df[column_name].min(), df[column_name].max()
        if thresholds[0] >= col_min:
            thresholds = [col_min - 1] + thresholds
        if thresholds[-1] < col_max:
            thresholds.append(col_max)
        df[categorical_name] = pd.cut(df[column_name], bins=thresholds)
    return df


def spread_categorical_missingness_to_numerical(deid_df: pd.DataFrame,
                                                numerical_quasi_identifiers: List[str]) -> pd.DataFrame:
    """
    input: de-identified dataframe (output of min_missingness_max_deid or min_suppr_max_deid),
    list of the numerical quasi-identifiers that were converted with convert_numerical_to_categorical
    blank the numerical value wherever its <column>_categorical version is missing
    output: copy of the dataframe with the numerical columns blanked and the <column>_categorical columns removed
    """
    out = deid_df.copy()
    for feature in numerical_quasi_identifiers:
        out.loc[out[feature + "_categorical"].isna(), feature] = np.nan
    return out.drop(columns=[feature + "_categorical" for feature in numerical_quasi_identifiers])
