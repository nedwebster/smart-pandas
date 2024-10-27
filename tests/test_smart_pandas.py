import pandas as pd


def test_config_ingestion(smart_data):

    assert smart_data.config.name == "life_expectancy_modelling_data"

    pd.testing.assert_frame_equal(
        smart_data.config.raw_features, smart_data[["weight", "height", "age"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.config.raw_features, smart_data[["weight", "height", "age"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.config.derived_features, smart_data[["bmi"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.config.model_features, smart_data[["age", "bmi"]]
    )

    pd.testing.assert_frame_equal(
        smart_data.config.target, smart_data[["life_expectancy"]]
    )
