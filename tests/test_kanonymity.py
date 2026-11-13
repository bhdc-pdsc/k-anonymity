import pytest

import kanonymity as ka
import pandas as pd



def test_k_anonymity():
    """
    Test that check_k-anonymity returns what's expected.
    """
    df = pd.DataFrame({
        "one":['A','A','A','A','B','B','B'],
        "two":['j','i','j','i','i','j','i'],
        "three":['z','z','z','z','z','z','z'],
    })

    num_violating, summary_df = ka.check_k_anonymity(df, ['one', 'two'], 2)

    assert num_violating==1


def test_k_anonymity_counts_missing_as_any_category():
    """
    Test that a missing value matches every category.
    """
    df = pd.DataFrame({"a": ["x", "x", "y", None], "b": ["p", "p", "p", "p"]})
    num_violating, summary_df = ka.check_k_anonymity(df, ["a", "b"], k=3)
    # rows 0,1 match each other + row 3 -> 3; row 2 matches itself + row 3 -> 2; row 3 matches everything -> 4
    counts = dict(zip(summary_df["a"].fillna("NA"), summary_df["wildcard_count"]))
    assert counts == {"x": 3, "y": 2, "NA": 4}
    assert num_violating == 1


def test_k_anonymity_rejects_numerical_quasi_identifier():
    """
    Test that a numerical quasi-identifier raises an error.
    """
    df = pd.DataFrame({"age": [20, 30, 40], "sex": ["M", "F", "M"]})
    with pytest.raises(ValueError, match="categorical"):
        ka.check_k_anonymity(df, ["age", "sex"], 2)


def test_k_anonymity_ignores_numerical_column_not_in_quasi_identifiers():
    """
    Test that a numerical column that is not a quasi-identifier doesn't raise an error.
    """
    df = pd.DataFrame({"age": [20, 30, 40], "sex": ["M", "F", "M"]})
    num_violating, _ = ka.check_k_anonymity(df, ["sex"], 2)
    assert num_violating == 1



def test_safe_combinations():
    """
    Test that get safe combinations works. 
    """
    df = pd.DataFrame({
        "one":['A','A','A','A','B','B','B'],
        "two":['j','i','j','i','i','j','i'],
        "three":['z','z','z','z','z','z','z'],
    })

    valid_combi = ka.safe_combinations(df, ['one', 'two', 'three'], 2)

    assert type(valid_combi) is list


def test_min_missingness_max_deid():
    """
    Test if the addition of missingness works.
    """
    df = pd.DataFrame({
        "one":['A','A','A','A','B','B','B'],
        "two":['j','i','j','i','i','j','i'],
        "three":['z','z','z','z','z','z','z'],
    })

    best_df, best_missingness, best_violating = ka.min_missingness_max_deid(df, ['one', 'two', 'three'], 2, maximum_missingness=25, excluded_features=None, step=0.001, n_random_trials=10)

    assert best_violating==0
    assert isinstance(best_df, pd.DataFrame)