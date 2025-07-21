import pytest

from smart_pandas import pandas as pd


@pytest.fixture
def column_tags():
    from smart_pandas.config.tag import TAG_CONFIGS, create_tag

    return [create_tag(tag_name) for tag_name in TAG_CONFIGS.keys()]


@pytest.fixture()
def smart_data_raw():
    smart_data = pd.DataFrame(
        {
            "user_id": ["1", "2", "3"],
            "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "name": ["Ned", "Roland", "Tom"],
            "weight": [78, 74, 80],
            "height": [180, 182, 185],
            "age": [31, 31, 34],
            "life_expectancy": [80, 80, 80],
        }
    )
    smart_data.smart_pandas.load_config(config_path="tests/example_configs/example_config.yaml")
    return smart_data


@pytest.fixture()
def smart_data_processed(smart_data_raw):
    smart_data_processed = smart_data_raw.copy()
    smart_data_processed.smart_pandas.load_config(config_path="tests/example_configs/example_config.yaml")
    smart_data_processed.loc[:, "bmi"] = smart_data_processed["weight"] / (smart_data_processed["height"] / 100) ** 2
    return smart_data_processed


@pytest.fixture
def load_config():
    from smart_pandas.config.config_utils import read_config

    def _loader(path):
        return read_config(f"tests/example_configs/{path}.yaml")
    return _loader
