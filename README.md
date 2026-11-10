# k-anonymity package

Evaluate if a tabular dataset respect k-anonymity handling missing values in the dataset. K-anonymity stands for having at least k similar rows for each row in the dataset, including itself. A missing value has value of any other category, having missing values increase the deidentification. This package includes a function to add the minimum number of missing values to reach k-anonymity of the dataset.

## Installation

```bash
uv add kanonymity
```

## Important information

The current version of the pipeline does not directly handle longitudinal data. You should take care of deciding how to consider duplicates subjects. One example here, where it is decided to keep most common answer, but you could also decide to replace answers that changes accross one subject by missing value, or consider the different rows as different individuals:

```python
# df is the original complete dataframe
# replace "subject_id" by the name of the ID column
# quasi_identifiers is a list of quasi-identifiers column names
# index_table helps to reduplicate rows in the end in case some modification of the dataframe is done
table = df.groupby(["subject_id"]+quasi_identifiers, dropna=False).size().to_frame('size').reset_index()
index_table = df.groupby(['subject_id'], dropna=False).apply(lambda x: x.index.tolist()).to_frame('original_index').reset_index()
index_table = index_table.sort_values(['subject_id'], ascending=[True])
df = table.sort_values(["subject_id", "size"], ascending=[True, False]).drop_duplicates(subset=["subject_id"], keep="first")
df = df.drop('size', axis=1)
```

Here is how to reduplicate the subjects that had several rows in the original dataset:

```python
# df is the new version of the deduplicated dataset that had potential modifications to increase deidentification
# original_df is the dataframe before these potential modifications
for r in range(len(df)):
    if not original_df.iloc[r].equals(df.iloc[r]):
        indexes = index_table['original_index'].iloc[r]
        for i in indexes:
            for feature in quasi_identifiers:
                if feature.split('_')[0] not in numerical_QI:
                    flat_df.loc[i, feature] = df[feature].iloc[r]
                else:
                    if original_df[feature].iloc[r] != df[feature].iloc[r]:
                        flat_df[feature.split('_')[0]].iloc[i] = np.nan
```

In addition, the quasi-identifiers have to be categorical columns. You can use relevant bins to convert numerical to categorical.

## Functions details

- `check_k_anonymity(df, quasi_identifier_cols, k)` returns the number of rows that don't have enough similar rows, and are at risk of reidentification. The columns to considere are given in a list `quasi_identifers_cols` and the number of similar rows to have, including the row itself is `k`. It also returns a summary df of every combination of each categories combinations that exists in the dataset. Missing values are counting as if they could be any existing category. If a feature `A` can have `0` or `1` as categories, a row with missing `A` would count as two rows, one with `0` and one with `1`.
- `safe_combinations(df, quasi_identifiers, k, min_combi_size=2, max_combi_size=None)` checks which combinations of features validates k-anonymity. In case you have a large number of quasi-identifiers, you can set boundaries to the number of features to consider in your combinations. The result is given as a list of lists.
- `min_missingness_max_deid(df, QI, k, maximum_missingness=25, excluded_features=None, step=0.001, n_random_trials=25)` adds the same percentage of missing values in every quasi-identifier feature, with a maximum of `maximum_missingness` to reach no risk of reidentification. Binary search is implemented to find the lowest percentage that allows no risk of reidentification, with several random trials as the position of the missing values impacts the k-anonymity. The step defines the precision of the percentage and the function allows to exclude some columns from the missingness addition. If the `maximum_missingness` threshold doesn't allow to earase the risk of reidentification, it returns the dataframe with the minimum number of rows at risk of reidentification. The function returns the most deidentified dataframe with the minimum addition of missingness possible, the percentage of missingness added to each column and the number of rows that are at risk of reidentification.

## De-identification pipeline and minimal suppression

These functions cover the full de-identification of a tabular dataset (CSV or Parquet), from raw data to the table to release:

1. **Remove direct identifiers** such as names and contact details.
2. **Coarsen dates** to the year.
3. **k-anonymity**: turn numerical quasi-identifiers into categories, then blank as few cells as possible until every row matches at least `k` rows. Missing values count as wildcards, as in `check_k_anonymity`.

```python
import kanonymity as ka

df = ka.load_data("file.csv")

# Step 1: direct identifiers (names not in the data are ignored)
df = ka.remove_direct_identifiers(df, ["Name", "Surname", "email address", "phone number", "address"])

# Step 2: dates -> years (date columns must have a datetime dtype)
df = ka.simplify_dates(df)

# Step 3: k-anonymity
quasi_identifiers = ["Age", "Diagnosis", "Sex"]
ka.describe_numerical(df, quasi_identifiers)        # min / max / mean of numerical ones

df = ka.convert_numerical_to_categorical(df, "Age", bin_size=3)   # 3 equal-width age groups
# or: ka.convert_numerical_to_categorical(df, "Age", categories_thresholds=[30, 45])
quasi_identifiers = ka.replace_with_categorical(df, quasi_identifiers)

new_df, blanked_cells = ka.minimal_suppression(df, quasi_identifiers, k=5, maximum_percentage=25)

release_df = ka.build_release(new_df, blanked_cells)  # blanked cells -> "missing"
ka.save_data(release_df, "file_modified.csv")
```

- `load_data(file_path)` / `save_data(df, file_path)` read and write CSV or Parquet, chosen from the file extension (Parquet needs `pip install "kanonymity[parquet]"`).
- `remove_direct_identifiers(df, identifiers)` drops the identifier columns present in the data.
- `simplify_dates(df)` keeps only the year of every datetime column.
- `describe_numerical(df, columns)` / `find_numerical_columns(df, columns)` list the numerical quasi-identifiers that must be categorized.
- `convert_numerical_to_categorical(df, column_name, bin_size=None, categories_thresholds=None)` adds a `<column>_categorical` column. `bin_size` is the **number** of equal-width bins, not their width; `categories_thresholds` gives the bin edges instead.
- `replace_with_categorical(df, quasi_identifiers)` swaps numerical quasi-identifiers for their `_categorical` versions.
- `check_wildcard_k_anonymity(df, quasi_identifier_cols, k)` is the same check as `check_k_anonymity`, but only requires the quasi-identifier columns to be categorical.
- `minimal_suppression(df, quasi_identifier_cols, k, maximum_percentage=25, verbose=True)` blanks one cell at a time, each time the cell that fixes the most rows at risk, until k-anonymity is reached or a column would exceed `maximum_percentage` missing values (values already missing count too). It then puts back any blanked cell that is not needed. Unlike `min_missingness_max_deid`, which blanks the same random percentage in every column, it only blanks the cells that are needed, so it usually removes fewer values, but its run time grows quickly with the number of rows: it suits datasets of up to a few hundred rows. It returns the new dataframe and the list of blanked cells with their original values.
- `build_release(new_df, blanked_cells, missing_label="missing")` drops the `_categorical` helper columns, keeps the exact original values, and writes `missing_label` in the blanked cells. Note that k-anonymity was checked on the categories, not on the exact values.
