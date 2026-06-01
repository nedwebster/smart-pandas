"""Example: defining a SmartPandas config as a Python object instead of YAML."""

import pandas as pd

from smart_pandas import pandas as spd
from smart_pandas.config.data_config import DataConfig
from smart_pandas.smart_dict import SmartDict


# Define the config as a Python object — same structure as the YAML,
# but using dicts and lists directly.
config = DataConfig(
    name="life_expectancy_modelling_data",
    columns=[
        {
            "name": "user_id",
            "data_schema": {"dtype": "str"},
            "tags": ["unique_identifier"],
        },
        {
            "name": "timestamp",
            "data_schema": {"dtype": "datetime"},
            "tags": ["row_timestamp"],
        },
        {
            "name": "name",
            "data_schema": {"dtype": "str"},
            "tags": ["metadata"],
        },
        {
            "name": "weight",
            "data_schema": {"dtype": "float"},
            "tags": ["raw_feature"],
        },
        {
            "name": "height",
            "data_schema": {"dtype": "float"},
            "tags": ["raw_feature"],
        },
        {
            "name": "age",
            "data_schema": {
                "raw": {"dtype": "str"},
                "processed": {"dtype": "int"},
            },
            "tags": ["raw_feature", "model_feature"],
        },
        {
            "name": "bmi",
            "data_schema": {"dtype": "float"},
            "tags": ["derived_feature", "model_feature"],
        },
        {
            "name": "life_expectancy",
            "data_schema": {"dtype": "int"},
            "tags": ["target"],
        },
    ],
)


# Use with SmartPandas — pass config= instead of config_path=
df = spd.DataFrame(
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
df.smart_pandas.load_config(config=config)
print(f"SmartPandas state: {df.smart_pandas.state}")
df.smart_pandas.validate()


# Use with SmartDict — same config object
record = SmartDict(
    {
        "user_id": "1",
        "timestamp": pd.Timestamp("2020-01-01"),
        "name": "Ned",
        "weight": 78.0,
        "height": 180.0,
        "age": "31",
        "life_expectancy": 80,
    },
    config=config,
)
print(f"SmartDict state: {record.state}")
record.validate()
print(f"Raw features: {record.raw_features}")
