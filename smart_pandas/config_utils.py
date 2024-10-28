from typing import List
import yaml
import pandera as pa

from smart_pandas.column import Column
from smart_pandas.column_set import ColumnSet
from smart_pandas.label import Label, LABEL_MAP
from smart_pandas.label_set import LabelSet
from smart_pandas.data_config import DataConfig


def read_yaml(path: str):
    with open(path) as stream:
        return yaml.safe_load(stream)


def build_label(name: str) -> Label:
    return LABEL_MAP[name]()


def build_label_set(label_strings: List[str]) -> LabelSet:
    return LabelSet(labels=[build_label(label) for label in label_strings])


def build_column(
    name: str, schema: dict, labels: List[str], description: str
) -> Column:
    return Column(
        name=name,
        schema=pa.Column(**schema),
        labels=build_label_set(labels),
        description=description,
    )


def build_column_set(column_data: List[dict]) -> ColumnSet:
    return ColumnSet(columns=[build_column(**column) for column in column_data])


def read_config(path: str) -> ColumnSet:
    config = read_yaml(path)
    name = config["name"]
    column_set = build_column_set(config["columns"])

    return DataConfig(name=name, column_set=column_set)
