import pandera as pa
from pydantic import BaseModel
from smart_pandas.column_set import ColumnSet
from smart_pandas.schema_utils import default_df_schema_params


class DataConfig(BaseModel):
    """
    Config object for a dataset.

    The config allows validation of column names and types, as well as attributes for extracting specific columns.

    Parameters
    ----------
    name: str
        The name of the dataset.
    column_set: ColumnSet
        The column set of the dataset.
    """
    name: str
    column_set: ColumnSet

    @property
    def raw_features(self):
        return [
            column.name for column in self.column_set
            if any(tag.name == "raw_feature" for tag in column.tags)
        ]

    @property
    def derived_features(self):
        return [
            column.name
            for column in self.column_set
            if any(tag.name == "derived_feature" for tag in column.tags)
        ]

    @property
    def model_features(self):
        return [
            column.name
            for column in self.column_set
            if any(tag.name == "model_feature" for tag in column.tags)
        ]

    @property
    def target(self):
        return [
            column.name for column in self.column_set
            if any(tag.name == "target" for tag in column.tags)
        ] or None

    @property
    def unique_identifier(self):
        return [
            column.name for column in self.column_set
            if any(tag.name == "unique_identifier" for tag in column.tags)
        ] or None

    @property
    def metadata(self):
        return [
            column.name for column in self.column_set
            if any(tag.name == "metadata" for tag in column.tags)
        ] or None

    @property
    def row_timestamp(self):
        return [
            column.name for column in self.column_set
            if any(tag.name == "row_timestamp" for tag in column.tags)
        ] or None

    @property
    def schema(self) -> pa.DataFrameSchema:
        column_schemas = {column.name: column.schema for column in self.column_set}
        return pa.DataFrameSchema(column_schemas, **default_df_schema_params())
