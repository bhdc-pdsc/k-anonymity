import numpy as np
import pandas as pd
import pytest

import kanonymity as ka


def synthetic_df(n=40, seed=0):
    """Small fake dataset (no real participants)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "participant_id": [f"P{i:03d}" for i in range(n)],
            "Sex": rng.choice(["Male", "Female"], n, p=[0.7, 0.3]),
            "Diagnosis": rng.choice(["AUD", "CUD", "AUD+CUD"], n, p=[0.5, 0.35, 0.15]),
            "Age": rng.integers(19, 61, n),
        }
    )


def categorized_df():
    """synthetic_df with Age converted to 3 age groups."""
    return ka.convert_numerical_to_categorical(synthetic_df(), "Age", bin_size=3)


QI = ["Age_categorical", "Diagnosis", "Sex"]


def capped_df():
    """
    Rows 8 (y, q) and 9 (z, q) have no match. Blanking column a in one of them fixes both,
    but column a is already 30% missing.
    """
    return pd.DataFrame(
        {
            "a": [None, None, None, "x", "x", "x", "x", "x", "y", "z"],
            "b": ["p"] * 8 + ["q", "q"],
        }
    )


# --- numerical to categorical --------------------------------------------

def test_bin_size_is_number_of_bins():
    df = categorized_df()
    assert df["Age_categorical"].nunique() <= 3
    assert df["Age_categorical"].notna().all()


def test_convert_does_not_modify_input():
    df = synthetic_df()
    original = df.copy()
    out = ka.convert_numerical_to_categorical(df, "Age", bin_size=3)
    pd.testing.assert_frame_equal(df, original)
    assert "Age_categorical" in out.columns


def test_categories_thresholds_used_as_edges():
    df = pd.DataFrame({"Age": [19, 30, 31, 45, 46, 60]})
    out = ka.convert_numerical_to_categorical(df, "Age", categories_thresholds=[45, 30])
    cats = out["Age_categorical"].astype(str).tolist()
    assert cats == ["(18, 30]", "(18, 30]", "(30, 45]", "(30, 45]", "(45, 60]", "(45, 60]"]


def test_categories_thresholds_with_missing_values():
    df = pd.DataFrame({"Age": [19.0, np.nan, 50.0]})
    out = ka.convert_numerical_to_categorical(df, "Age", categories_thresholds=[30])
    assert out["Age_categorical"].notna().tolist() == [True, False, True]


@pytest.mark.parametrize("kwargs", [{}, {"bin_size": 3, "categories_thresholds": [30]}])
def test_convert_requires_exactly_one_option(kwargs):
    with pytest.raises(ValueError):
        ka.convert_numerical_to_categorical(synthetic_df(), "Age", **kwargs)


def test_spread_categorical_missingness_to_numerical():
    deid_df = pd.DataFrame(
        {
            "Age": [40.0, 50.0, np.nan, 33.0],
            "Age_categorical": ["(30, 45]", None, None, "(30, 45]"],  # row 1 blanked, row 2 was already missing
            "Sex": ["Male", "Female", "Male", None],
        }
    )
    original = deid_df.copy()
    out = ka.spread_categorical_missingness_to_numerical(deid_df, ["Age"])
    assert list(out.columns) == ["Age", "Sex"]
    assert out["Age"].isna().tolist() == [False, True, True, False]
    assert out["Age"].iloc[[0, 3]].tolist() == [40.0, 33.0]
    pd.testing.assert_frame_equal(deid_df, original)  # input not modified


# --- min_suppr_max_deid -------------------------------------------------

def test_min_suppr_max_deid_reaches_k_anonymity():
    df = categorized_df()
    original = df.copy()

    new_df, added, n_violating = ka.min_suppr_max_deid(df, QI, k=3, maximum_total_missingness=50, verbose=False)

    assert n_violating == 0
    assert ka.check_k_anonymity(new_df, QI, 3)[0] == 0
    pd.testing.assert_frame_equal(df, original)  # input not modified
    assert set(added) == set(QI)
    for col in QI:
        expected = (new_df[col].isna().sum() - original[col].isna().sum()) * 100 / len(df)
        assert added[col] == pytest.approx(expected)
        assert new_df[col].isna().mean() <= 0.5
    # non-quasi-identifier columns are never touched
    pd.testing.assert_frame_equal(new_df[["participant_id", "Age"]], original[["participant_id", "Age"]])


def test_min_suppr_max_deid_already_anonymous():
    df = pd.DataFrame({"a": ["x"] * 4})
    new_df, added, n_violating = ka.min_suppr_max_deid(df, ["a"], k=2, verbose=False)
    assert added == {"a": 0.0}
    assert n_violating == 0
    pd.testing.assert_frame_equal(new_df, df)


def test_min_suppr_max_deid_excluded_features_untouched():
    df = categorized_df()
    new_df, added, _ = ka.min_suppr_max_deid(
        df, QI, k=3, maximum_total_missingness=50, excluded_features=["Sex"], verbose=False
    )
    pd.testing.assert_series_equal(new_df["Sex"], df["Sex"])
    assert added["Sex"] == 0.0


def test_min_suppr_max_deid_excluded_feature_not_in_qi():
    with pytest.raises(ValueError, match="not in quasi-identifiers"):
        ka.min_suppr_max_deid(categorized_df(), QI, k=3, excluded_features=["participant_id"], verbose=False)


def test_min_suppr_max_deid_cap_counts_existing_missingness():
    """Column a is already 30% missing: a 25% cap allows nothing more, a 40% cap allows one more cell."""
    df = capped_df()

    # cap reached: best result returned, nothing blanked, rows still at risk
    new_df, added, n_violating = ka.min_suppr_max_deid(
        df, ["a", "b"], k=2, maximum_total_missingness=25, excluded_features=["b"], verbose=False
    )
    assert n_violating == 2
    pd.testing.assert_frame_equal(new_df, df)
    assert added == {"a": 0.0, "b": 0.0}

    # 40% total allows one more missing value in a, which is enough
    new_df, added, n_violating = ka.min_suppr_max_deid(
        df, ["a", "b"], k=2, maximum_total_missingness=40, excluded_features=["b"], verbose=False
    )
    assert n_violating == 0
    assert new_df["a"].isna().sum() == 4
    assert added == {"a": 10.0, "b": 0.0}


def test_min_suppr_max_deid_returns_best_result_when_cap_reached():
    df = categorized_df()
    new_df, added, n_violating = ka.min_suppr_max_deid(df, QI, k=10, maximum_total_missingness=5, verbose=False)
    assert isinstance(new_df, pd.DataFrame)
    assert n_violating == ka.check_k_anonymity(new_df, QI, 10)[0]
    assert n_violating <= ka.check_k_anonymity(df, QI, 10)[0]
    for col in QI:
        assert new_df[col].isna().sum() <= int(len(df) * 5 / 100)
