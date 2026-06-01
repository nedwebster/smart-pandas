import pytest

import pandas as pd
from smart_pandas.state import State, StateName, MLStage, StateError


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


def test_smart_pandas_auto_update(smart_data_raw):
    smart_data_raw.smart_pandas.auto_update = False
    assert smart_data_raw.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)

    # validate the state has not automatically updated
    smart_data_raw.loc[:, "bmi"] = smart_data_raw["weight"] / (smart_data_raw["height"] / 100) ** 2
    assert smart_data_raw.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING) 

    smart_data_raw.smart_pandas.update()
    assert smart_data_raw.smart_pandas.state == State(name=StateName.PROCESSED, ml_stage=MLStage.TRAINING)


def test_validate(smart_data_raw):
    smart_data_raw.smart_pandas.validate()


def test_smart_pandas_pinned_state(smart_data_processed):
    """Passing state='raw' should override inference."""
    smart_data_processed.smart_pandas.load_config(
        config_path="tests/example_configs/example_config.yaml",
        state="raw",
    )
    assert smart_data_processed.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)


def test_smart_pandas_pinned_state_survives_column_change(smart_data_raw):
    """Pinned state should not change when columns change."""
    smart_data_raw.smart_pandas.load_config(
        config_path="tests/example_configs/example_config.yaml",
        state="raw",
    )
    assert smart_data_raw.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)

    # Adding bmi would normally transition to PROCESSED
    smart_data_raw.loc[:, "bmi"] = smart_data_raw["weight"] / (smart_data_raw["height"] / 100) ** 2
    assert smart_data_raw.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)


def test_smart_pandas_set_state(smart_data_raw):
    """set_state should change the pinned state and rebuild schema."""
    smart_data_raw.smart_pandas.load_config(config_path="tests/example_configs/example_config.yaml")
    assert smart_data_raw.smart_pandas.state.name == StateName.RAW

    smart_data_raw.smart_pandas.set_state("processed")
    assert smart_data_raw.smart_pandas.state.name == StateName.PROCESSED


def test_smart_pandas_set_state_none_re_enables_inference(smart_data_processed):
    """set_state(None) should clear the pin and re-enable inference."""
    smart_data_processed.smart_pandas.set_state("raw")
    assert smart_data_processed.smart_pandas.state.name == StateName.RAW

    smart_data_processed.smart_pandas.set_state(None)
    assert smart_data_processed.smart_pandas.state.name == StateName.PROCESSED


def test_validate_inplace(smart_data_raw):
    smart_data_raw.smart_pandas.validate(inplace=True)
    assert smart_data_raw["weight"].dtype == float


def test_validate_returns_dataframe(smart_data_raw):
    result = smart_data_raw.smart_pandas.validate(inplace=False)
    assert isinstance(result, pd.DataFrame)
    assert result.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)


def test_access_before_load_config():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = df.smart_pandas.state


def test_incompatible_attribute_returns_none(smart_data_raw):
    """model_features should be None in RAW state."""
    assert smart_data_raw.smart_pandas.model_features is None


def test_corrupted_state_on_missing_required_column(smart_data_raw):
    smart_data_raw.drop(columns=["user_id"], inplace=True)
    assert smart_data_raw.smart_pandas.state.name == StateName.CORRUPTED


def test_validate_corrupted_raises(smart_data_raw):
    smart_data_raw.drop(columns=["user_id"], inplace=True)
    with pytest.raises(StateError):
        smart_data_raw.smart_pandas.validate()


def test_inference_state(smart_data_processed):
    """Dropping target should give INFERENCE ml_stage."""
    smart_data_processed.drop(columns=["life_expectancy"], inplace=True)
    smart_data_processed.smart_pandas.load_config(config_path="tests/example_configs/example_config.yaml")
    assert smart_data_processed.smart_pandas.state.ml_stage == MLStage.INFERENCE
