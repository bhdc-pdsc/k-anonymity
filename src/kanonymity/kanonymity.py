from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def check_k_anonymity(df: pd.DataFrame, quasi_identifier_cols: List[str], k: int) -> Tuple[int, pd.DataFrame]:
    """
    input: dataframe, list of quasi identifiers columns, number of rows that should have the same carateristics to respect k-anonymity
    takes missing value as it can be any other category
    output: number of rows that don't belong to a group of at least k rows, dataframe with the different combination of quasi-identifiers 
    """
    sub_df = df[quasi_identifier_cols].copy()
    nas = sub_df.isna().to_numpy()

    numerical_col = sub_df.select_dtypes(include=np.number).columns.tolist()
    if numerical_col:
        raise ValueError(f"Numerical columns: {numerical_col} have to be converted to categorical.")

    vals = sub_df.astype(object).to_numpy().copy()
    vals[nas] = "__WILDCARD_TMP_VAL__"
    val_equal = vals[:, None, :] == vals[None, :, :]

    either_na = nas[:, None, :] | nas[None, :, :]
    row_col_matches = val_equal | either_na

    match_matrix = row_col_matches.all(axis=-1)
    row_counts = match_matrix.sum(axis=1)

    is_safe_row = row_counts >= k
    num_violating_rows = int((~is_safe_row).sum())

    results_df = sub_df.copy()
    results_df["wildcard_count"] = row_counts
    results_df["is_safe"] = is_safe_row

    summary_df = results_df.drop_duplicates().reset_index(drop=True)

    return num_violating_rows, summary_df




def safe_combinations(df: pd.DataFrame, quasi_identifiers: List[str], k: int, min_combi_size: int = 2,
                      max_combi_size: Optional[int] = None) -> List[List[str]]:
    """
    input dataframe, list of quasi-identifiers, number of rows that should have the same carateristics to respect k-anonymity, minimum and maximum number of features for each combinations
    output a list of list of features combinations that respect the k-anonymity
    """
    if max_combi_size is None:
        max_combi_size = len(quasi_identifiers)
    combis = [comb for i in range(1, len(quasi_identifiers)+1) for comb in combinations(quasi_identifiers, i)]
    valid_features_combination = []
    for c in combis:
        if len(c)>=min_combi_size and len(c)<=max_combi_size:
            num_violating_rows, _ = check_k_anonymity(df, list(c), k)
            if num_violating_rows==0:
                valid_features_combination.append(list(c))
    return valid_features_combination



def min_missingness_max_deid(df: pd.DataFrame, QI: List[str], k: int, maximum_missingness: float = 25,
                             excluded_features: Optional[List[str]] = None, step: float = 0.001,
                             n_random_trials: int = 25) -> Tuple[pd.DataFrame, float, int]:
    """
    input: dataframe, list of quasi-identifiers, number of similar rows in the dataset to ensure no risk of reidentification, 
    maximum percentage of missing data to add in each QI feature, list of QI features that should be excluded from information suppression,
    precision percentage to search (step), number of random missingness addition the largest the table and number of QI, the biggest it should be
    return: dataframe with minimum missingness for maximum deidentification, percentage of missingness removed, number of violating rows
    """
    np.random.seed(42)

    if excluded_features is None:
        excluded_features = []

    original_df = df.copy()

    # Check excluded features
    for i in excluded_features:
        if i not in QI:
            raise ValueError(f"Excluded feature '{i}' not in quasi-identifiers.")
    considered_QI = [col for col in QI if col not in excluded_features]

    # Check if maximum missingness exceeds available non-null values
    for f in considered_QI:
        percent_missing = df[f].isnull().sum() * 100.0 / len(df)
        if maximum_missingness > (100.0 - percent_missing):
            raise ValueError(f"Maximum missingness is higher than percentage of available non-null values in '{f}'.")

    # Test feasibility at maximum_missingness first
    max_num_to_add = int(round((maximum_missingness / 100.0) * len(df)))
    max_violating = float('inf')
    for _ in range(n_random_trials):
        test_df = original_df.copy()
        for col in considered_QI:
            non_missing_idx = test_df[test_df[col].notna()].index
            if len(non_missing_idx) >= max_num_to_add:
                target_idx = np.random.choice(non_missing_idx, size=max_num_to_add, replace=False)
                test_df.loc[target_idx, col] = np.nan
        violating_rows, _ = check_k_anonymity(test_df, QI, k)
        if violating_rows < max_violating:
            max_violating = violating_rows
            max_df = test_df
            if max_violating == 0:
                break

    # If maximum missingness cannot achieve k-anonymity
    if max_violating > 0:
        print(f"Maximum missingness ({maximum_missingness}%) does not achieve k-anonymity. Returning best de-identification possible (violating rows: {max_violating}).")
        return max_df, maximum_missingness, max_violating

    # Binary search between 0% and maximum_missingness to find the minimum required percentage
    low = 0.0
    high = float(maximum_missingness)
    best_missingness = high
    best_df = max_df
    best_violating = 0

    while (high - low) >= step:
        current_missingness = round((low + high) / 2.0, 4)
        
        current_num_to_add = int(round((current_missingness / 100.0) * len(df)))
        min_local_violating = float('inf')

        # Try a few random attempts for this percentage
        for _ in range(n_random_trials):
            test_df = original_df.copy()
            for col in considered_QI:
                non_missing_idx = test_df[test_df[col].notna()].index
                if len(non_missing_idx) >= current_num_to_add:
                    target_idx = np.random.choice(non_missing_idx, size=current_num_to_add, replace=False)
                    test_df.loc[target_idx, col] = np.nan
            
            violating_rows, _ = check_k_anonymity(test_df, QI, k)
            if violating_rows < min_local_violating:
                min_local_violating = violating_rows
                current_df = test_df
                if min_local_violating == 0:
                    break
        current_violating = min_local_violating

        # Update binary search bounds based on whether k-anonymity is achieved
        if current_violating == 0:
            # k-anonymity achieved, save result and try a smaller percentage
            best_missingness = current_missingness
            best_df = current_df
            best_violating = 0
            high = current_missingness - step
        else:
            # Failed k-anonymity, we need more missingness
            low = current_missingness + step

    return best_df, best_missingness, best_violating


def min_suppr_max_deid(df: pd.DataFrame, quasi_identifier_cols: List[str], k: int,
                       maximum_total_missingness: float = 25, excluded_features: Optional[List[str]] = None,
                       verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, float], int]:
    """
    input: dataframe, list of quasi identifiers columns, number of rows that should have the same carateristics to respect k-anonymity,
    maximum percentage of missing values allowed in each column (values already missing count too),
    list of quasi-identifiers that should never be blanked
    blank (set to missing) as few cells as possible so that every row has at least k matches, a missing value can be any other category
    any row can be blanked, not only the violating ones
    if k-anonymity can't be reached within maximum_total_missingness, the best de-identified dataframe found is returned
    output: dataframe with blanked cells, percentage of missingness added in each quasi-identifier column, number of violating rows
    """
    if excluded_features is None:
        excluded_features = []

    # Check excluded features
    for i in excluded_features:
        if i not in quasi_identifier_cols:
            raise ValueError(f"Excluded feature '{i}' not in quasi-identifiers.")
    considered_cols = [col for col in quasi_identifier_cols if col not in excluded_features]

    new_df = df.copy()
    blanked_cells = []
    # maximum number of missing cells allowed per column (rounded down)
    max_missing = int(len(df) * maximum_total_missingness / 100)

    # starting number of rows that don't have at least k matches
    # (excluded columns are never blanked but still count for re-identification)
    num_violating_rows, _ = check_k_anonymity(new_df, quasi_identifier_cols, k)

    # each round blanks ONE cell: the one that fixes the most violating rows
    while num_violating_rows > 0:
        # try blanking every cell of every row, on a copy, and keep the best one
        best_cell, best_score, best_violations = None, None, None
        for row in new_df.index:
            for col in considered_cols:
                if pd.isna(new_df.at[row, col]):  # already missing, nothing to blank
                    continue
                if new_df[col].isna().sum() >= max_missing:  # column reached the missingness cap
                    continue
                test_df = new_df.copy()
                test_df.at[row, col] = np.nan
                test_violations, _ = check_k_anonymity(test_df, quasi_identifier_cols, k)
                score = num_violating_rows - test_violations  # number of violating rows this blank fixes
                if best_score is None or score > best_score:
                    best_cell, best_score, best_violations = (row, col), score, test_violations

        # no cell could be tested: all are already missing or their column is at the cap
        if best_cell is None:
            if verbose:
                print(
                    f"k-anonymity can't be reached within {maximum_total_missingness}% missing values per column. "
                    f"Returning the best de-identification found ({num_violating_rows} rows still at risk)."
                )
            break

        # blank the best cell for real (save the original value first)
        row, col = best_cell
        blanked_cells.append({"row": row, "column": col, "original_value": new_df.at[row, col]})
        new_df.at[row, col] = np.nan
        num_violating_rows = best_violations

    # put back the blanked cells that turn out not to be needed
    for cell in list(blanked_cells):
        test_df = new_df.copy()
        test_df.at[cell["row"], cell["column"]] = cell["original_value"]
        test_violations, _ = check_k_anonymity(test_df, quasi_identifier_cols, k)
        if test_violations <= num_violating_rows:
            new_df = test_df
            num_violating_rows = test_violations
            blanked_cells.remove(cell)

    # percentage of missing values added in each quasi-identifier column
    added_missingness = {
        col: float((new_df[col].isna().sum() - df[col].isna().sum()) * 100 / len(df))
        for col in quasi_identifier_cols
    }

    if verbose:
        for col in quasi_identifier_cols:
            print(f"{col}: {added_missingness[col]:.1f}% missing added, {new_df[col].isna().mean() * 100:.1f}% missing in total")

    return new_df, added_missingness, num_violating_rows
