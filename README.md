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