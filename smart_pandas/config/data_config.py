from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from smart_pandas.config.column_set import ColumnSet


class DataConfig(BaseModel):
    """
    Config object for a dataset.

    The config allows validation of column names and types, as well as attributes for extracting specific columns.

    Parameters
    ----------
    name: str
        The name of the dataset.
    columns: ColumnSet
        The column set of the dataset.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    name: str
    columns: ColumnSet

    @field_validator("columns", mode="before")
    def parse_column_set(cls, v):
        if isinstance(v, list):
            return ColumnSet(columns=v)
        return v

    @computed_field
    def raw_features(self) -> list[str]:
        return [
            column.name for column in self.columns
            if any(tag.name == "raw_feature" for tag in column.tags)
        ]

    @computed_field
    @property
    def derived_features(self) -> list[str]:
        return [
            column.name
            for column in self.columns
            if any(tag.name == "derived_feature" for tag in column.tags)
        ]

    @computed_field
    @property
    def model_features(self) -> list[str]:
        return [
            column.name
            for column in self.columns
            if any(tag.name == "model_feature" for tag in column.tags)
        ]

    @computed_field
    @property
    def target(self) -> list[str]:
        return [
            column.name for column in self.columns
            if any(tag.name == "target" for tag in column.tags)
        ]

    @computed_field
    @property
    def unique_identifier(self) -> list[str]:
        return [
            column.name for column in self.columns
            if any(tag.name == "unique_identifier" for tag in column.tags)
        ]

    @computed_field
    @property
    def metadata(self) -> list[str]:
        return [
            column.name for column in self.columns
            if any(tag.name == "metadata" for tag in column.tags)
        ]

    @computed_field
    @property
    def row_timestamp(self) -> list[str]:
        return [
            column.name for column in self.columns
            if any(tag.name == "row_timestamp" for tag in column.tags)
        ]
