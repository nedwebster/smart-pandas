import pandas as pd
import pytest

from smart_pandas import pandas as spd
from smart_pandas.config.column import Column
from smart_pandas.state import State, StateName, MLStage


@pytest.fixture()
def smart_data_raw_str_age():
    """Raw data where age is a string (state-dependent type config)."""
    data = spd.DataFrame(
        {
            "user_id": ["1", "2", "3"],
            "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "name": ["Ned", "Roland", "Tom"],
            "weight": [78.0, 74.0, 80.0],
            "height": [180.0, 182.0, 185.0],
            "age": ["31", "31", "34"],
            "life_expectancy": [80, 80, 80],
        }
    )
    data.smart_pandas.load_config(config_path="tests/example_configs/state_dependent_types_config.yaml")
    return data


@pytest.fixture()
def smart_data_processed_int_age(smart_data_raw_str_age):
    """Processed data where age has been converted to int."""
    data = smart_data_raw_str_age.copy()
    data.smart_pandas.load_config(config_path="tests/example_configs/state_dependent_types_config.yaml")
    data["age"] = data["age"].astype(int)
    data.loc[:, "bmi"] = data["weight"] / (data["height"] / 100) ** 2
    return data


def test_state_dependent_schema_raw(smart_data_raw_str_age):
    """Age column should validate as str in RAW state."""
    assert smart_data_raw_str_age.smart_pandas.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)
    smart_data_raw_str_age.smart_pandas.validate()


def test_state_dependent_schema_processed(smart_data_processed_int_age):
    """Age column should validate as int in PROCESSED state."""
    assert smart_data_processed_int_age.smart_pandas.state == State(name=StateName.PROCESSED, ml_stage=MLStage.TRAINING)
    smart_data_processed_int_age.smart_pandas.validate()


def test_state_dependent_column_get_data_schema():
    """get_data_schema returns the correct schema per state."""
    col = Column(
        name="age",
        data_schema={"raw": {"dtype": "str"}, "processed": {"dtype": "int"}},
        tags=["raw_feature", "model_feature"],
    )
    raw_schema = col.get_data_schema(StateName.RAW)
    processed_schema = col.get_data_schema(StateName.PROCESSED)

    assert "str" in raw_schema.dtype.type.name
    assert "int" in processed_schema.dtype.type.name


def test_state_dependent_column_fallback():
    """get_data_schema falls back to first entry for unlisted states."""
    col = Column(
        name="age",
        data_schema={"raw": {"dtype": "str"}, "processed": {"dtype": "int"}},
        tags=["raw_feature", "model_feature"],
    )
    # UNKNOWN is not in the dict, should fall back to first entry ("raw")
    fallback_schema = col.get_data_schema(StateName.UNKNOWN)
    assert "str" in fallback_schema.dtype.type.name


def test_single_schema_column_unchanged():
    """Single-schema columns still work via get_data_schema."""
    col = Column(
        name="weight",
        data_schema={"dtype": "float"},
        tags=["raw_feature"],
    )
    schema = col.get_data_schema(StateName.RAW)
    assert "float" in schema.dtype.type.name

    schema_processed = col.get_data_schema(StateName.PROCESSED)
    assert "float" in schema_processed.dtype.type.name
