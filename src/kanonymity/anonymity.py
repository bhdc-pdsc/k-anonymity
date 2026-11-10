import numpy as np
import pandas as pd

_WILDCARD_PLACEHOLDER = "__WILDCARD_TMP_VAL__"


def check_wildcard_k_anonymity(df, quasi_identifier_cols, k):
    """
    input: dataframe, list of quasi identifiers columns, number of rows that should have the same carateristics to respect k-anonymity
    takes missing value as it can be any other category
    output: number of rows that don't belong to a group of at least k rows, dataframe with the different combination of quasi-identifiers
    """
    numerical_col = df[quasi_identifier_cols].select_dtypes(include=np.number).columns.tolist()
    if numerical_col:
        raise ValueError(
            f"Numerical columns: {numerical_col} have to be converted to categorical. "
            "Use convert_numerical_to_categorical function."
        )

    sub_df = df[quasi_identifier_cols].copy()
    nas = sub_df.isna().to_numpy()

    vals = sub_df.astype(object).to_numpy().copy()
    vals[nas] = _WILDCARD_PLACEHOLDER

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


def minimal_suppression(df, quasi_identifier_cols, k, maximum_percentage=25, verbose=True):
    """
    input: dataframe, list of quasi identifiers columns, number of rows that should have the same carateristics to respect k-anonymity, maximum percentage of missing values allowed per column
    blank (set to missing) as few cells as possible so that every row has at least k matches, a missing value can be any other category
    any row can be blanked, not only the violating ones
    a column is never blanked beyond maximum_percentage of missing values (values already missing count too)
    output: dataframe with blanked cells, dataframe with the blanked cells and their original values
    """
    new_df = df.copy()
    blanked_cells = []
    # maximum number of missing cells allowed per column (rounded down)
    max_missing = int(len(df) * maximum_percentage / 100)

    # starting number of rows that don't have at least k matches
    num_violating_rows, _ = check_wildcard_k_anonymity(new_df, quasi_identifier_cols, k)

    # each round blanks ONE cell: the one that fixes the most violating rows
    while num_violating_rows > 0:
        # try blanking every cell of every row, on a copy, and keep the best one
        best_cell, best_score, best_violations = None, None, None
        for row in new_df.index:
            for col in quasi_identifier_cols:
                if pd.isna(new_df.at[row, col]):  # already missing, nothing to blank
                    continue
                if new_df[col].isna().sum() >= max_missing:  # column reached the missingness cap
                    continue
                test_df = new_df.copy()
                test_df.at[row, col] = np.nan
                test_violations, _ = check_wildcard_k_anonymity(test_df, quasi_identifier_cols, k)
                score = num_violating_rows - test_violations  # number of violating rows this blank fixes
                if best_score is None or score > best_score:
                    best_cell, best_score, best_violations = (row, col), score, test_violations

        # no cell could be tested: all are already missing or their column is at the cap
        if best_cell is None:
            if verbose:
                print(
                    f"Stopped: no more cells can be blanked within {maximum_percentage}% missing "
                    f"per column, {num_violating_rows} rows still don't have enough matches."
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
        test_violations, _ = check_wildcard_k_anonymity(test_df, quasi_identifier_cols, k)
        if test_violations <= num_violating_rows:
            new_df = test_df
            blanked_cells.remove(cell)

    # missing values per column after blanking
    if verbose:
        for col in quasi_identifier_cols:
            print(f"{col}: {new_df[col].isna().mean() * 100:.1f}% missing")

    return new_df, pd.DataFrame(blanked_cells, columns=["row", "column", "original_value"])
