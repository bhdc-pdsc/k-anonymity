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