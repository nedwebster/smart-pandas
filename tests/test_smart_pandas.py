import pandas as pd
from smart_pandas.state import State, StateName, MLStage


def test_smart_pandas(smart_data_raw):

    assert smart_data_raw.smart_pandas.name == "life_expectancy_modelling_data"

    pd.testing.assert_frame_equal(
        smart_data_raw.smart_pandas.raw_features, smart_data_raw[["weight", "height", "age"]]
    )

    pd.testing.assert_frame_equal(
        smart_data_raw.smart_pandas.raw_features, smart_data_raw[["weight", "height", "age"]]
    )

    pd.testing.assert_frame_equal(
        smart_data_raw.smart_pandas.target, smart_data_raw[["life_expectancy"]]
    )

    assert smart_data_raw.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)

def test_smart_pandas_processed(smart_data_processed):

    assert smart_data_processed.smart_pandas.state == State(name=StateName.PROCESSED, ml_stage=MLStage.TRAINING)

    pd.testing.assert_frame_equal(
        smart_data_processed.smart_pandas.derived_features, smart_data_processed[["bmi"]]
    )

    pd.testing.assert_frame_equal(
        smart_data_processed.smart_pandas.model_features, smart_data_processed[["age", "bmi"]]
    )
