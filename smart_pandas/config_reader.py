from typing import List
import yaml

from smart_pandas.column import Column
from smart_pandas.column_set import ColumnSet
from smart_pandas.label import Label, LABEL_MAP
from smart_pandas.label_set import LabelSet
from smart_pandas.dataframe import SmartDataFrame


class ConfigReader:
    def __init__(self):
        pass

    @staticmethod
    def read_yaml(path: str):
        with open(path) as stream:
            return yaml.safe_load(stream)

    @staticmethod
    def build_label(name: str) -> Label:
        return LABEL_MAP[name]()

    def build_label_set(self, label_strings: List[str]) -> LabelSet:
        return LabelSet(labels=[self.build_label(label) for label in label_strings])

    def build_column(
        self, name: str, type: str, labels: List[str], description: str
    ) -> Column:
        return Column(
            name=name,
            type=type,
            labels=self.build_label_set(labels),
            description=description,
        )

    def build_column_set(self, column_data: List[dict]) -> ColumnSet:
        return ColumnSet(
            columns=[self.build_column(**column) for column in column_data]
        )

    def read_config(self, path: str) -> ColumnSet:
        config = self.read_yaml(path)
        name = config["name"]
        column_set = self.build_column_set(config["columns"])

        return SmartDataFrame(name=name, column_set=column_set)
