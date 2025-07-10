from typing import List
import yaml
import pandera as pa

from smart_pandas.column import Column
from smart_pandas.column_set import ColumnSet
from smart_pandas.tag import Tag, TAG_MAP
from smart_pandas.tag_set import TagSet
from smart_pandas.data_config import DataConfig


def read_yaml(path: str):
    with open(path) as stream:
        return yaml.safe_load(stream)


def build_tag(name: str) -> Tag:
    return TAG_MAP[name]()


def build_tag_set(tag_strings: List[str]) -> TagSet:
    return TagSet(tags=[build_tag(tag) for tag in tag_strings])


def build_column(
    name: str, schema: dict, tags: List[str], description: str
) -> Column:
    return Column(
        name=name,
        schema=pa.Column(**schema),
        tags=build_tag_set(tags),
        description=description,
    )


def build_column_set(column_data: List[dict]) -> ColumnSet:
    return ColumnSet(columns=[build_column(**column) for column in column_data])


def read_config(path: str) -> DataConfig:
    config = read_yaml(path)
    name = config["name"]
    column_set = build_column_set(config["columns"])

    return DataConfig(
        name=name, column_set=column_set
    )
