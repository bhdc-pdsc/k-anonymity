import numpy as np
import pandas as pd

DEFAULT_DIRECT_IDENTIFIERS = ["Name", "Surname", "email address", "phone number", "address"]


def remove_direct_identifiers(df, identifiers=DEFAULT_DIRECT_IDENTIFIERS):
    """
    input: dataframe, list of the direct identifiers
    output: dataframe without the direct identifiers
    """
    present = [c for c in identifiers if c in df.columns]
    if not present:
        print("No direct identifiers found in the dataset.")
        return df.copy()
    print(f"Removed direct identifiers: {present}")
    return df.drop(columns=present)


def simplify_dates(df):
    """
    input: dataframe, date columns should be in datetime format
    output: dataframe with years only for dates
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.year
    return df


def find_numerical_columns(df, columns):
    """
    input: dataframe, list of quasi-identifiers
    output: list of the numerical quasi-identifiers
    """
    numeric = set(df[columns].select_dtypes(include=np.number).columns)
    return [c for c in columns if c in numeric]


def describe_numerical(df, columns):
    """
    input: dataframe, list of quasi-identifiers
    prints min, max and mean of the numerical quasi-identifiers
    output: list of the numerical quasi-identifiers
    """
    numerical = find_numerical_columns(df, columns)
    if not numerical:
        print("There is no numerical feature in the provided quasi-identifiers list.")
        return numerical
    print(f"There are {len(numerical)} numerical feature(s):")
    for col in numerical:
        print(col, df[col].min(), df[col].max(), df[col].mean())
    return numerical


def convert_numerical_to_categorical(df, column_name, bin_size=None, categories_thresholds=None):
    """
    input: dataframe, name of the numerical column to convert to categorical, number of bins for categorization or list of categories thresholds
    create categorical version based either on the number of bins or thresholds defined in a list
    output: dataframe with the new column <column_name>_categorical
    """
    if bin_size and categories_thresholds:
        raise ValueError("Both bin_size and categories_thresholds are provided, please provide only one.")
    if not bin_size and not categories_thresholds:
        raise ValueError("No bin_size or categories_thresholds provided, please provide one (but only one).")

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


def replace_with_categorical(df, quasi_identifiers):
    """
    input: dataframe, list of quasi-identifiers
    output: list of quasi-identifiers where numerical ones are replaced by their categorical version
    """
    numerical = find_numerical_columns(df, quasi_identifiers)
    for col in numerical:
        if col + "_categorical" not in df.columns:
            raise ValueError(
                f"Column {col} has not been categorized. Use convert_numerical_to_categorical first."
            )
    return [c + "_categorical" if c in numerical else c for c in quasi_identifiers]
