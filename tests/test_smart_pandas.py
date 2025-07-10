import pandas as pd


def test_config_ingestion(smart_data):

    assert smart_data.smart_pandas.name == "life_expectancy_modelling_data"

    pd.testing.assert_frame_equal(
        smart_data.smart_pandas.raw_features, smart_data[["weight", "height", "age"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.smart_pandas.raw_features, smart_data[["weight", "height", "age"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.smart_pandas.derived_features, smart_data[["bmi"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.smart_pandas.model_features, smart_data[["age", "bmi"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.smart_pandas.target, smart_data[["life_expectancy"]]
    )
