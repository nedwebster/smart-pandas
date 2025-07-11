import yaml
from smart_pandas.config.data_config import DataConfig


def read_yaml(path: str):
    with open(path) as stream:
        return yaml.safe_load(stream)


def read_config(path: str) -> DataConfig:
    config_data = read_yaml(path)
    return DataConfig(**config_data)
