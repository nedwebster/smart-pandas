import pytest

import pandas as pd

from smart_pandas.smart_dict import SmartDict, DictValidationError
from smart_pandas.state import State, StateName, MLStage, StateError


CONFIG_PATH = "tests/example_configs/example_config.yaml"
STATE_DEPENDENT_CONFIG_PATH = "tests/example_configs/state_dependent_types_config.yaml"


@pytest.fixture
def raw_record():
    return {
        "user_id": "1",
        "timestamp": pd.Timestamp("2020-01-01"),
        "name": "Ned",
        "weight": 78.0,
        "height": 180.0,
        "age": 31,
        "life_expectancy": 80,
    }


@pytest.fixture
def processed_record():
    return {
        "user_id": "1",
        "timestamp": pd.Timestamp("2020-01-01"),
        "name": "Ned",
        "weight": 78.0,
        "height": 180.0,
        "age": 31,
        "bmi": 24.1,
        "life_expectancy": 80,
    }


@pytest.fixture
def inference_record():
    """Processed record without target — typical inference input."""
    return {
        "user_id": "1",
        "timestamp": pd.Timestamp("2020-01-01"),
        "name": "Ned",
        "weight": 78.0,
        "height": 180.0,
        "age": 31,
        "bmi": 24.1,
    }


def test_smart_dict_raw_state(raw_record):
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert sd.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)


def test_smart_dict_processed_state(processed_record):
    sd = SmartDict(processed_record, config_path=CONFIG_PATH)
    assert sd.state == State(name=StateName.PROCESSED, ml_stage=MLStage.TRAINING)


def test_smart_dict_inference_state(inference_record):
    sd = SmartDict(inference_record, config_path=CONFIG_PATH)
    assert sd.state == State(name=StateName.PROCESSED, ml_stage=MLStage.INFERENCE)


def test_smart_dict_validate_raw(raw_record):
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    result = sd.validate()
    assert isinstance(result, dict)
    assert result["user_id"] == "1"


def test_smart_dict_validate_processed(processed_record):
    sd = SmartDict(processed_record, config_path=CONFIG_PATH)
    result = sd.validate()
    assert isinstance(result, dict)


def test_smart_dict_tag_attributes(raw_record):
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert sd.raw_features == {"weight": 78.0, "height": 180.0, "age": 31}
    assert sd.target == {"life_expectancy": 80}
    assert sd.metadata == {"name": "Ned"}
    # model_features is incompatible with RAW state
    assert sd.model_features is None


def test_smart_dict_processed_tag_attributes(processed_record):
    sd = SmartDict(processed_record, config_path=CONFIG_PATH)
    assert sd.model_features == {"age": 31, "bmi": 24.1}
    assert sd.derived_features == {"bmi": 24.1}


def test_smart_dict_corrupted_state():
    """Missing required columns should give CORRUPTED state."""
    sd = SmartDict({"age": 31}, config_path=CONFIG_PATH)
    assert sd.state.name == StateName.CORRUPTED


def test_smart_dict_validate_corrupted_raises():
    sd = SmartDict({"age": 31}, config_path=CONFIG_PATH)
    with pytest.raises(StateError):
        sd.validate()


def test_smart_dict_requires_config():
    with pytest.raises(ValueError, match="Either config or config_path must be provided"):
        SmartDict({"age": 31})


def test_smart_dict_requires_dict():
    with pytest.raises(TypeError, match="Expected dict"):
        SmartDict([1, 2, 3], config_path=CONFIG_PATH)


def test_smart_dict_state_dependent_types():
    """State-dependent types should work: age as str in RAW, int in PROCESSED."""
    raw = SmartDict(
        {
            "user_id": "1",
            "timestamp": pd.Timestamp("2020-01-01"),
            "name": "Ned",
            "weight": 78.0,
            "height": 180.0,
            "age": "31",
            "life_expectancy": 80,
        },
        config_path=STATE_DEPENDENT_CONFIG_PATH,
    )
    assert raw.state.name == StateName.RAW
    raw.validate()

    processed = SmartDict(
        {
            "user_id": "1",
            "timestamp": pd.Timestamp("2020-01-01"),
            "name": "Ned",
            "weight": 78.0,
            "height": 180.0,
            "age": 31,
            "bmi": 24.1,
            "life_expectancy": 80,
        },
        config_path=STATE_DEPENDENT_CONFIG_PATH,
    )
    assert processed.state.name == StateName.PROCESSED
    processed.validate()


def test_smart_dict_wrong_type_raises(raw_record):
    """Validation should fail when a value has the wrong type."""
    raw_record["age"] = "thirty-one"  # str instead of int
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    with pytest.raises(DictValidationError, match="age"):
        sd.validate()


def test_smart_dict_extra_keys_raises(raw_record):
    """Validation should fail when unexpected keys are present."""
    raw_record["extra_column"] = 42
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    with pytest.raises(DictValidationError, match="extra_column"):
        sd.validate()


def test_smart_dict_nullable(raw_record):
    """None values should be rejected for non-nullable columns."""
    raw_record["age"] = None
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    with pytest.raises(DictValidationError, match="age"):
        sd.validate()


def test_smart_dict_does_not_mutate(raw_record):
    """Validate should not mutate the original dict."""
    original = raw_record.copy()
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    sd.validate()
    assert raw_record == original


def test_smart_dict_setitem(raw_record):
    """Assignment should update the data and trigger state re-inference."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert sd.state == State(name=StateName.RAW, ml_stage=MLStage.TRAINING)

    sd["bmi"] = 24.1
    assert sd["bmi"] == 24.1
    assert sd.state == State(name=StateName.PROCESSED, ml_stage=MLStage.TRAINING)


def test_smart_dict_getitem(raw_record):
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert sd["age"] == 31
    assert sd["user_id"] == "1"


def test_smart_dict_delitem(raw_record):
    """Deleting a required column should move to CORRUPTED state."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    del sd["user_id"]
    assert "user_id" not in sd
    assert sd.state.name == StateName.CORRUPTED


def test_smart_dict_contains(raw_record):
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert "age" in sd
    assert "nonexistent" not in sd


def test_smart_dict_is_a_dict(raw_record):
    """SmartDict should be a real dict — isinstance, len, iteration all work."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert isinstance(sd, dict)
    assert len(sd) == len(raw_record)
    assert set(sd.keys()) == set(raw_record.keys())
    assert list(sd.values()) == list(raw_record.values())
    assert dict(sd) == raw_record


def test_smart_dict_update_method(raw_record):
    """dict.update() should trigger state re-inference."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert sd.state.name == StateName.RAW
    sd.update({"bmi": 24.1})
    assert sd["bmi"] == 24.1
    assert sd.state.name == StateName.PROCESSED


def test_smart_dict_pop(raw_record):
    """dict.pop() should trigger state re-inference."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    val = sd.pop("user_id")
    assert val == "1"
    assert "user_id" not in sd
    assert sd.state.name == StateName.CORRUPTED


def test_smart_dict_pinned_state_string(processed_record):
    """Passing state='raw' should override inference."""
    sd = SmartDict(processed_record, config_path=CONFIG_PATH, state="raw")
    assert sd.state.name == StateName.RAW


def test_smart_dict_pinned_state_enum(processed_record):
    """Passing state=StateName.RAW should override inference."""
    sd = SmartDict(processed_record, config_path=CONFIG_PATH, state=StateName.RAW)
    assert sd.state.name == StateName.RAW


def test_smart_dict_pinned_state_survives_mutation(raw_record):
    """Pinned state should not change when keys change."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH, state="raw")
    assert sd.state.name == StateName.RAW
    sd["bmi"] = 24.1
    # Without pinning this would become PROCESSED
    assert sd.state.name == StateName.RAW


def test_smart_dict_pinned_state_validates_with_correct_schema():
    """Pinned state should use the corresponding schema for the pinned state.

    With the state-dependent config, age is str in RAW and int in PROCESSED.
    Both records have the same columns — only the type of age differs.
    Without pinning, both would infer as PROCESSED (model_features present).
    """
    raw_record = {
        "user_id": "1",
        "timestamp": pd.Timestamp("2020-01-01"),
        "name": "Ned",
        "weight": 78.0,
        "height": 180.0,
        "age": "31",
        "life_expectancy": 80,
    }
    # Without pinning, inference picks RAW here (no bmi/derived_features).
    # Pin to RAW explicitly — age validates as str.
    sd_raw = SmartDict(raw_record, config_path=STATE_DEPENDENT_CONFIG_PATH, state="raw")
    assert sd_raw.state.name == StateName.RAW
    sd_raw.validate()

    processed_record = {
        "user_id": "1",
        "timestamp": pd.Timestamp("2020-01-01"),
        "name": "Ned",
        "weight": 78.0,
        "height": 180.0,
        "age": 31,
        "bmi": 24.1,
        "life_expectancy": 80,
    }
    # Pin to PROCESSED — age validates as int.
    sd_proc = SmartDict(processed_record, config_path=STATE_DEPENDENT_CONFIG_PATH, state="processed")
    assert sd_proc.state.name == StateName.PROCESSED
    sd_proc.validate()


def test_smart_dict_validate_inplace_coerces(raw_record):
    """inplace=True should write coerced values back into the SmartDict."""
    # age is int(31), config expects int, but weight is int(78) and config expects float
    raw_record["weight"] = 78  # int, should be coerced to float
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert isinstance(sd["weight"], int)

    result = sd.validate(inplace=True)
    assert result is None
    assert isinstance(sd["weight"], float)
    assert sd["weight"] == 78.0


def test_smart_dict_validate_inplace_returns_none(raw_record):
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert sd.validate(inplace=True) is None


def test_smart_dict_validate_not_inplace_returns_dict(raw_record):
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    result = sd.validate(inplace=False)
    assert isinstance(result, dict)
    assert not isinstance(result, SmartDict)


def test_smart_dict_set_state(raw_record):
    """set_state should change the pinned state and rebuild schema."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH)
    assert sd.state.name == StateName.RAW

    sd.set_state("processed")
    assert sd.state.name == StateName.PROCESSED

    sd.set_state(StateName.RAW)
    assert sd.state.name == StateName.RAW


def test_smart_dict_set_state_none_re_enables_inference(raw_record):
    """set_state(None) should clear the pin and re-enable inference."""
    sd = SmartDict(raw_record, config_path=CONFIG_PATH, state="processed")
    assert sd.state.name == StateName.PROCESSED

    sd.set_state(None)
    # Inference should now pick RAW based on the data
    assert sd.state.name == StateName.RAW
