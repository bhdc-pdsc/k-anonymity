# k-anonymity package

Evaluate if a tabular dataset respect k-anonymity handling missing values in the dataset.

## Installation

```bash
uv add kanonymity
```

or 

```bash
pip install kanonymity
```

## What is k-anonymity

K-anonymity is a data deidentification methods that counts the number of similar rows in a tabular dataset. To reach k-anonymity, there should be at least k similar rows in the dataset, including itself. The verification is done over the the quasi-identifier feature combinations present in the dataset. A missing value has value of any other category, having missing values increase the deidentification. There are several ways to reach k-anonymity:
- removing some quasi-identifiers columns. The function `safe_combinations` can help making decisions of removing features.
- removing some individuals at risk of reidentification
- merging categories to make broader groups
- adding missingness. The package contains two functions to add missingness to your dataset.

## Methods to add missing values and limit the risk of reidentification

Both functions blank (set to missing) some quasi-identifier values until every row has at least `k` similar rows, and both count a missing value as any other category. If k-anonymity can't be reached within the allowed missingness, both return the most deidentified dataframe they found. They differ in **which** values they blank and in how the maximum missingness is defined:

| | `min_missingness_max_deid` | `min_suppr_max_deid` |
|---|---|---|
| Which values are blanked | Random values, the **same percentage in every** quasi-identifier column | One value at a time, each time the one that fixes the most rows at risk |
| Number of values blanked | Usually more, since there is a limited attempts of random trials | Usually fewer |
| Maximum missingness | `maximum_missingness`: percentage of missing values **added** to each column, on top of the values already missing | `maximum_total_missingness`: percentage of missing values in each column **in total**, values already missing count too |
| Speed | Faster, but longer if you increase the number of random trials | Slower, especially with bigger datasets and when there is a lot of risk of reidentification |
| Result | Same result on every run (fixed seed) | Same result on every run |
| Returns | dataframe, percentage of missingness added (same for every column), number of rows at risk | dataframe, percentage of missingness added in each column, number of rows at risk |

**Which one to choose**

- `min_missingness_max_deid` for large datasets, or when the released data will be analysed and the missing values should be spread at random so that they don't bias a subgroup.
- `min_suppr_max_deid` for small datasets, when you want to remove as few values as possible.

You can run both on the same data and compare the number of blanked values.

## Handling longitudinal data

The package does not handle longitudinal data. You should take care of deciding how to consider duplicates subjects. One example here, where it is decided to keep most common answer, but you could also decide to replace answers that changes accross one subject by missing value, or consider the different rows as different individuals, among a lot of other solutions:

```python
# df is the original complete dataframe
# replace "subject_id" by the name of the ID column
# quasi_identifiers is a list of quasi-identifiers column names
# index_table helps to reduplicate rows in the end in case some modification of the dataframe is done
table = df.groupby(["subject_id"]+quasi_identifiers, dropna=False).size().to_frame('size').reset_index()
index_table = df.groupby(['subject_id'], dropna=False).apply(lambda x: x.index.tolist()).to_frame('original_index').reset_index()
index_table = index_table.sort_values(['subject_id'], ascending=[True]).reset_index(drop=True)
dedup_df = table.sort_values(["subject_id", "size"], ascending=[True, False]).drop_duplicates(subset=["subject_id"], keep="first")
dedup_df = dedup_df.drop('size', axis=1).reset_index(drop=True)
```

Here is how to reduplicate the subjects that had several rows in the original dataset:

```python
# deid_df is dedup_df after the modifications to increase deidentification (same rows in the same order)
# flat_df is the full dataframe with the modifications copied to every row of each subject
flat_df = df.copy()
for r in range(len(deid_df)):
    if not dedup_df.iloc[r].equals(deid_df.iloc[r]):
        indexes = index_table['original_index'].iloc[r]
        for i in indexes:
            for feature in quasi_identifiers:
                flat_df.loc[i, feature] = deid_df[feature].iloc[r]
```

## Handling numerical features

/!\ depending on what is decided about the numerical to categorical function or not, only mention that numerical should be categorized and then the missing values spread back to numerical features, if the solution of adding missing values is taken. Otherwise it is possible to make an example of how to use the functions if we find it is relevant.




## Functions description


- `check_k_anonymity(df, quasi_identifier_cols, k)` returns the number of rows that don't have enough similar rows, and are at risk of reidentification. The columns to considere are given in a list `quasi_identifers_cols` and the number of similar rows to have, including the row itself is `k`. It also returns a summary df of every combination of each categories combinations that exists in the dataset. Missing values are counting as if they could be any existing category. If a feature `A` can have `0` or `1` as categories, a row with missing `A` would count as two rows, one with `0` and one with `1`. The quasi-identifiers must be categorical, other columns of the dataframe can be numerical.
- `safe_combinations(df, quasi_identifiers, k, min_combi_size=2, max_combi_size=None)` checks which combinations of features validates k-anonymity. In case you have a large number of quasi-identifiers, you can set boundaries to the number of features to consider in your combinations. The result is given as a list of lists.
- `min_missingness_max_deid(df, QI, k, maximum_missingness=25, excluded_features=None, step=0.001, n_random_trials=25)` adds the same percentage of missing values in every quasi-identifier feature, with a maximum of `maximum_missingness` to reach no risk of reidentification. Binary search is implemented to find the lowest percentage that allows no risk of reidentification, with several random trials as the position of the missing values impacts the k-anonymity. The step defines the precision of the percentage and the function allows to exclude some columns from the missingness addition. If the `maximum_missingness` threshold doesn't allow to erase the risk of reidentification, it returns the dataframe with the minimum number of rows at risk of reidentification. The function returns the most deidentified dataframe with the minimum addition of missingness possible, the percentage of missingness added to each column and the number of rows that are at risk of reidentification.
- `min_suppr_max_deid(df, quasi_identifier_cols, k, maximum_total_missingness=25, excluded_features=None, verbose=True)` blanks one cell at a time, each time the cell that fixes the most rows at risk, until k-anonymity is reached or no more cell can be blanked without a column exceeding `maximum_total_missingness` missing values in total (values already missing count too). It then puts back any blanked cell that is not needed. The columns in `excluded_features` are never blanked but still count in the k-anonymity check. If the `maximum_total_missingness` threshold doesn't allow to erase the risk of reidentification, it returns the dataframe with the minimum number of rows at risk of reidentification it found. The function returns the deidentified dataframe, a dictionary with the percentage of missingness added to each quasi-identifier column and the number of rows that are at risk of reidentification.
- `convert_numerical_to_categorical(df, column_name, bin_size=None, categories_thresholds=None)` returns a copy of the dataframe with a new `<column>_categorical` column. `bin_size` is the **number** of equal-width bins, not their width; `categories_thresholds` gives the bin edges instead.
- `spread_categorical_missingness_to_numerical(deid_df, numerical_quasi_identifiers)` blanks the numerical value wherever its `<column>_categorical` version is missing after adding missingness, and removes the `<column>_categorical` columns.
