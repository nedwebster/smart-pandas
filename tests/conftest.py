import pytest

from smart_pandas import pandas as pd


@pytest.fixture
def column_tags():
    from smart_pandas.config.tag import TAG_CONFIGS, create_tag

    return [create_tag(tag_name) for tag_name in TAG_CONFIGS.keys()]


@pytest.fixture
def smart_data():
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
    smart_data.smart_pandas.init(config_path="examples/example_config.yaml")
    smart_data["bmi"] = smart_data["weight"] / (smart_data["height"] / 100) ** 2
    return smart_data
