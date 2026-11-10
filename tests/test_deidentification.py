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


# --- io -----------------------------------------------------------------

def test_load_save_roundtrip_csv(tmp_path):
    df = synthetic_df(5)
    ka.save_data(df, tmp_path / "x.csv")
    pd.testing.assert_frame_equal(ka.load_data(tmp_path / "x.csv"), df, check_dtype=False)


def test_load_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        ka.load_data(tmp_path / "x.xlsx")


# --- step 1: direct identifiers ------------------------------------------

def test_remove_direct_identifiers_drops_present_columns():
    df = synthetic_df(5).assign(Name="a", **{"email address": "b"})
    out = ka.remove_direct_identifiers(df, ["Name", "email address", "phone number"])
    assert "Name" not in out.columns and "email address" not in out.columns
    assert len(out) == len(df)  # rows untouched
    assert "Name" in df.columns  # input not modified


def test_remove_direct_identifiers_none_present_returns_df():
    df = synthetic_df(5)
    out = ka.remove_direct_identifiers(df, ["Name"])
    pd.testing.assert_frame_equal(out, df)


# --- step 2: dates -------------------------------------------------------

@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_simplify_dates_any_datetime_resolution(unit):
    dates = pd.to_datetime(["2020-03-01", "2021-07-15"]).astype(f"datetime64[{unit}]")
    out = ka.simplify_dates(pd.DataFrame({"d": dates, "x": [1, 2]}))
    assert out["d"].tolist() == [2020, 2021]


def test_simplify_dates_timezone_aware():
    dates = pd.to_datetime(["2020-03-01", "2021-07-15"]).tz_localize("UTC")
    out = ka.simplify_dates(pd.DataFrame({"d": dates}))
    assert out["d"].tolist() == [2020, 2021]


# --- step 3: categorization ----------------------------------------------

def test_find_numerical_columns_includes_int32():
    df = pd.DataFrame({"a": np.array([1, 2], dtype="int32"), "b": ["x", "y"]})
    assert ka.find_numerical_columns(df, ["a", "b"]) == ["a"]


def test_bin_size_is_number_of_bins():
    df = synthetic_df()
    ka.convert_numerical_to_categorical(df, "Age", bin_size=3)
    assert df["Age_categorical"].nunique() <= 3
    assert df["Age_categorical"].notna().all()


def test_categories_thresholds_used_as_edges():
    df = pd.DataFrame({"Age": [19, 30, 31, 45, 46, 60]})
    ka.convert_numerical_to_categorical(df, "Age", categories_thresholds=[45, 30])
    cats = df["Age_categorical"].astype(str).tolist()
    assert cats == ["(18, 30]", "(18, 30]", "(30, 45]", "(30, 45]", "(45, 60]", "(45, 60]"]


def test_categories_thresholds_with_missing_values():
    df = pd.DataFrame({"Age": [19.0, np.nan, 50.0]})
    ka.convert_numerical_to_categorical(df, "Age", categories_thresholds=[30])
    assert df["Age_categorical"].notna().tolist() == [True, False, True]


@pytest.mark.parametrize("kwargs", [{}, {"bin_size": 3, "categories_thresholds": [30]}])
def test_convert_requires_exactly_one_option(kwargs):
    with pytest.raises(ValueError):
        ka.convert_numerical_to_categorical(synthetic_df(), "Age", **kwargs)


def test_replace_with_categorical():
    df = synthetic_df()
    with pytest.raises(ValueError, match="not been categorized"):
        ka.replace_with_categorical(df, ["Age", "Sex"])
    ka.convert_numerical_to_categorical(df, "Age", bin_size=3)
    assert ka.replace_with_categorical(df, ["Age", "Sex"]) == ["Age_categorical", "Sex"]


# --- k-anonymity ---------------------------------------------------------

def test_check_rejects_numerical():
    with pytest.raises(ValueError, match="categorical"):
        ka.check_wildcard_k_anonymity(synthetic_df(), ["Age"], 2)


def test_check_wildcard_counts():
    df = pd.DataFrame({"a": ["x", "x", "y", None], "b": ["p", "p", "p", "p"]})
    n_bad, summary = ka.check_wildcard_k_anonymity(df, ["a", "b"], k=3)
    # rows 0,1 match each other + row 3 (wildcard) -> 3; row 2 matches itself + row 3 -> 2;
    # row 3 matches everything -> 4
    counts = dict(zip(summary["a"].fillna("NA"), summary["wildcard_count"]))
    assert counts == {"x": 3, "y": 2, "NA": 4}
    assert n_bad == 1


def test_minimal_suppression_reaches_k_anonymity():
    df = synthetic_df()
    ka.convert_numerical_to_categorical(df, "Age", bin_size=3)
    qi = ka.replace_with_categorical(df, ["Age", "Diagnosis", "Sex"])
    original = df.copy()

    new_df, blanked = ka.minimal_suppression(df, qi, k=3, maximum_percentage=50, verbose=False)

    assert ka.check_wildcard_k_anonymity(new_df, qi, 3)[0] == 0
    pd.testing.assert_frame_equal(df, original)  # input not modified
    assert new_df[qi].isna().sum().sum() == len(blanked)
    for _, cell in blanked.iterrows():
        assert original.at[cell["row"], cell["column"]] == cell["original_value"]


def test_minimal_suppression_already_anonymous():
    df = pd.DataFrame({"a": ["x"] * 4})
    new_df, blanked = ka.minimal_suppression(df, ["a"], k=2, verbose=False)
    assert blanked.empty and list(blanked.columns) == ["row", "column", "original_value"]
    pd.testing.assert_frame_equal(new_df, df)


# --- release -------------------------------------------------------------

def test_build_release_keeps_exact_values_and_marks_blanked():
    new_df = pd.DataFrame(
        {
            "Sex": ["Male", None, "Female"],
            "Age": [40, 50, 33],
            "Age_categorical": ["(30, 45]", "(45, 60]", None],
            "Diagnosis": ["AUD", "CUD", None],  # row 2 Diagnosis was already missing
        }
    )
    blanked = pd.DataFrame(
        [
            {"row": 1, "column": "Sex", "original_value": "Male"},
            {"row": 2, "column": "Age_categorical", "original_value": "(30, 45]"},
        ]
    )
    out = ka.build_release(new_df, blanked)
    assert list(out.columns) == ["Sex", "Age", "Diagnosis"]
    assert out["Age"].tolist() == [40, 50, "missing"]
    assert out.at[1, "Sex"] == "missing"
    assert pd.isna(out.at[2, "Diagnosis"])
