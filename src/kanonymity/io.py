from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = (".csv", ".parquet", ".pq")


def load_data(file_path):
    """
    input: path to the dataset (csv or parquet)
    output: DataFrame
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix}. Use CSV or Parquet.")


def save_data(df, file_path):
    """
    input: dataframe, path of the file to write (csv or parquet)
    saves the dataframe without the index
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use CSV or Parquet.")
