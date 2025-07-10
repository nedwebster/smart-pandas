import pytest

from smart_pandas import pandas as pd


@pytest.fixture
def column_tags():
    from smart_pandas.tag import TAG_MAP

    return [value() for value in TAG_MAP.values()]


@pytest.fixture
def smart_data():
    smart_data = pd.DataFrame(
        {
            "user_id": ["1", "2", "3"],
            "name": ["Ned", "Roland", "Tom"],
            "weight": [78, 74, 80],
            "height": [180, 182, 185],
            "age": [31, 31, 34],
            "life_expectancy": [80, 80, 80],
        }
    )
    smart_data.config.init(config_path="examples/example_config.yaml")
    smart_data["bmi"] = smart_data["weight"] / (smart_data["height"] / 100) ** 2
    return smart_data
