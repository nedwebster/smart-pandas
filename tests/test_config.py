from smart_pandas.config.config_utils import read_config


def test_config_attributes():

    config = read_config("tests/example_configs/example_config.yaml")
    assert config.raw_features == ["weight", "height", "age"]
    assert config.derived_features == ["bmi"]
    assert config.model_features == ["age", "bmi"]
    assert config.target == ["life_expectancy"]
    assert config.unique_identifier == ["user_id"]
    assert config.metadata == ["name"]
    assert config.row_timestamp == ["timestamp"]
