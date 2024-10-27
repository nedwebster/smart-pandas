from pydantic import BaseModel
from smart_pandas.column_set import ColumnSet


class DataConfig(BaseModel):
    name: str
    column_set: ColumnSet

    @property
    def raw_features(self):
        return [
            column.name for column in self.column_set if "raw_feature" in column.labels
        ]

    @property
    def derived_features(self):
        return [
            column.name
            for column in self.column_set
            if "derived_feature" in column.labels
        ]

    @property
    def model_features(self):
        return [
            column.name
            for column in self.column_set
            if "model_feature" in column.labels
        ]

    @property
    def target(self):
        return [
            column.name for column in self.column_set if "target" in column.labels
        ] or None
