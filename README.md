# k-anonymity package

Evaluate if a tabular dataset respect k-anonymity handling missing values in the dataset. K-anonymity stands for having at least k similar rows for each row in the dataset, including itself. A missing value has value of any other category, having missing values increase the deidentification. This package includes functions to add missing values until the dataset reaches k-anonymity, in two different ways.

## Installation

```bash
uv add kanonymity
```

or `pip install kanonymity`. To read and write Parquet files, install `kanonymity[parquet]`.

## Two ways to reach k-anonymity

Both ways blank (set to missing) some quasi-identifier values until every row has at least `k` similar rows, and both count a missing value as any other category. They differ in **which** values they blank:

| | Way 1: `min_missingness_max_deid` | Way 2: `minimal_suppression` |
|---|---|---|
| Which values are blanked | Random values, the **same percentage in every** quasi-identifier column | Only the values needed: one at a time, each time the one that fixes the most rows at risk |
| Number of values blanked | Usually more, since columns that didn't need it are blanked too | Usually fewer |
| Where the missing values are | Spread at random over all rows | Concentrated in the rows with rare combinations of values |
| Speed | Fast, also on large datasets | Slow: suited to datasets of up to a few hundred rows |
| Missingness limit | `maximum_missingness`: maximum percentage **added** per column | `maximum_percentage`: maximum percentage missing **in total** per column, values already missing included |
| Result | Same random result on every run (fixed seed) | Same result on every run |

**Which one to choose**

- **Way 1** for large datasets, or when the released data will be analysed and the missing values should be spread at random so that they don't bias a subgroup.
- **Way 2** for small datasets, when you want to remove as few values as possible.

You can run both on the same data and compare the number of blanked values.

## Before you start (both ways)

The quasi-identifiers have to be categorical columns. You can use relevant bins to convert numerical to categorical, for example with `convert_numerical_to_categorical` (see [Preparing the data](#preparing-the-data)).

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

## Way 1: same percentage of missing values in every column

```python
import kanonymity as ka

quasi_identifiers = ["Age_categorical", "Diagnosis", "Sex"]

num_violating_rows, summary_df = ka.check_k_anonymity(df[quasi_identifiers], quasi_identifiers, k=5)

best_df, missingness_added, num_violating_rows = ka.min_missingness_max_deid(
    df[quasi_identifiers], quasi_identifiers, k=5, maximum_missingness=25
)
```

`check_k_anonymity` and `min_missingness_max_deid` require **every** column of the dataframe they receive to be categorical, which is why only the quasi-identifier columns are passed (`df[quasi_identifiers]`).

- `check_k_anonymity(df, quasi_identifier_cols, k)` returns the number of rows that don't have enough similar rows, and are at risk of reidentification. The columns to considere are given in a list `quasi_identifers_cols` and the number of similar rows to have, including the row itself is `k`. It also returns a summary df of every combination of each categories combinations that exists in the dataset. Missing values are counting as if they could be any existing category. If a feature `A` can have `0` or `1` as categories, a row with missing `A` would count as two rows, one with `0` and one with `1`.
- `safe_combinations(df, quasi_identifiers, k, min_combi_size=2, max_combi_size=None)` checks which combinations of features validates k-anonymity. In case you have a large number of quasi-identifiers, you can set boundaries to the number of features to consider in your combinations. The result is given as a list of lists.
- `min_missingness_max_deid(df, QI, k, maximum_missingness=25, excluded_features=None, step=0.001, n_random_trials=25)` adds the same percentage of missing values in every quasi-identifier feature, with a maximum of `maximum_missingness` to reach no risk of reidentification. Binary search is implemented to find the lowest percentage that allows no risk of reidentification, with several random trials as the position of the missing values impacts the k-anonymity. The step defines the precision of the percentage and the function allows to exclude some columns from the missingness addition. If the `maximum_missingness` threshold doesn't allow to earase the risk of reidentification, it returns the dataframe with the minimum number of rows at risk of reidentification. The function returns the most deidentified dataframe with the minimum addition of missingness possible, the percentage of missingness added to each column and the number of rows that are at risk of reidentification.

## Way 2: blank only the values needed

```python
import kanonymity as ka

quasi_identifiers = ["Age_categorical", "Diagnosis", "Sex"]

num_violating_rows, summary_df = ka.check_wildcard_k_anonymity(df, quasi_identifiers, k=5)

new_df, blanked_cells = ka.minimal_suppression(df, quasi_identifiers, k=5, maximum_percentage=25)

release_df = ka.build_release(new_df, blanked_cells)  # blanked cells -> "missing"
```

These functions only require the quasi-identifier columns to be categorical, so the full dataframe can be passed.

- `check_wildcard_k_anonymity(df, quasi_identifier_cols, k)` gives the same result as `check_k_anonymity`.
- `minimal_suppression(df, quasi_identifier_cols, k, maximum_percentage=25, verbose=True)` blanks one cell at a time, each time the cell that fixes the most rows at risk, until k-anonymity is reached or a column would exceed `maximum_percentage` missing values (values already missing count too). It then puts back any blanked cell that is not needed. It returns the new dataframe and the list of blanked cells with their original values.
- `build_release(new_df, blanked_cells, missing_label="missing")` drops the `_categorical` helper columns, keeps the exact original values, and writes `missing_label` in the blanked cells. Note that k-anonymity was checked on the categories, not on the exact values.

## Preparing the data

These functions can be used before either way. Full example, from the raw file to the file to release:

```python
import kanonymity as ka

df = ka.load_data("file.csv")

# Step 1: direct identifiers (names not in the data are ignored)
df = ka.remove_direct_identifiers(df, ["Name", "Surname", "email address", "phone number", "address"])

# Step 2: dates -> years (date columns must have a datetime dtype)
df = ka.simplify_dates(df)

# Step 3: quasi-identifiers must be categorical
quasi_identifiers = ["Age", "Diagnosis", "Sex"]
ka.describe_numerical(df, quasi_identifiers)        # min / max / mean of numerical ones

df = ka.convert_numerical_to_categorical(df, "Age", bin_size=3)   # 3 equal-width age groups
# or: ka.convert_numerical_to_categorical(df, "Age", categories_thresholds=[30, 45])
quasi_identifiers = ka.replace_with_categorical(df, quasi_identifiers)

# Step 4: Way 1 or Way 2 (here Way 2)
new_df, blanked_cells = ka.minimal_suppression(df, quasi_identifiers, k=5, maximum_percentage=25)
ka.save_data(ka.build_release(new_df, blanked_cells), "file_modified.csv")
```

- `load_data(file_path)` / `save_data(df, file_path)` read and write CSV or Parquet, chosen from the file extension (Parquet needs `kanonymity[parquet]`).
- `remove_direct_identifiers(df, identifiers)` drops the identifier columns present in the data.
- `simplify_dates(df)` keeps only the year of every datetime column.
- `describe_numerical(df, columns)` / `find_numerical_columns(df, columns)` list the numerical quasi-identifiers that must be categorized.
- `convert_numerical_to_categorical(df, column_name, bin_size=None, categories_thresholds=None)` adds a `<column>_categorical` column. `bin_size` is the **number** of equal-width bins, not their width; `categories_thresholds` gives the bin edges instead.
- `replace_with_categorical(df, quasi_identifiers)` swaps numerical quasi-identifiers for their `_categorical` versions.
