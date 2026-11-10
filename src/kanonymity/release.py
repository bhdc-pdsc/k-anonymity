CATEGORICAL_SUFFIX = "_categorical"


def build_release(new_df, blanked_cells, missing_label="missing"):
    """
    input: dataframe and blanked cells returned by minimal_suppression, text to write in the blanked cells
    keep the exact values instead of the categories (k-anonymity was checked on the categories)
    output: dataframe to release, with missing_label in the cells removed for de-identification
    """
    # categorical columns and the original column they were made from
    categorical_to_original = {
        col: col[: -len(CATEGORICAL_SUFFIX)]
        for col in new_df.columns
        if col.endswith(CATEGORICAL_SUFFIX) and col[: -len(CATEGORICAL_SUFFIX)] in new_df.columns
    }

    # astype(object) lets the text "missing" go into any column
    release_df = new_df.drop(columns=list(categorical_to_original)).astype(object)

    # write "missing" in the cells removed for de-identification
    # (a blanked age group means the exact age is removed)
    for _, cell in blanked_cells.iterrows():
        col = categorical_to_original.get(cell["column"], cell["column"])
        release_df.at[cell["row"], col] = missing_label

    return release_df
