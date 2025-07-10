from pydantic import BaseModel
from smart_pandas.column_set import ColumnSet


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
    def target(self) -> list[str]:
        return [
            column.name for column in self.column_set
            if any(tag.name == "target" for tag in column.tags)
        ]

    @property
    def unique_identifier(self) -> list[str]:
        return [
            column.name for column in self.column_set
            if any(tag.name == "unique_identifier" for tag in column.tags)
        ]

    @property
    def metadata(self) -> list[str]:
        return [
            column.name for column in self.column_set
            if any(tag.name == "metadata" for tag in column.tags)
        ]

    @property
    def row_timestamp(self) -> list[str]:
        return [
            column.name for column in self.column_set
            if any(tag.name == "row_timestamp" for tag in column.tags)
        ]
