import pandera as pa
from pydantic import BaseModel
from smart_pandas.column_set import ColumnSet
from smart_pandas.schema_utils import default_df_schema_params


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

    @property
    def schema(self) -> pa.DataFrameSchema:
        column_schemas = {column.name: column.schema for column in self.column_set}
        return pa.DataFrameSchema(column_schemas, **default_df_schema_params())
