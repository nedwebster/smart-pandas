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

    def _get_columns_by_tag(self, tag_name: str) -> list[str]:
        """Helper method to get column names by tag."""
        return [
            column.name for column in self.columns
            if any(tag.name == tag_name for tag in column.tags)
        ]

    @computed_field
    def raw_features(self) -> list[str]:
        return self._get_columns_by_tag("raw_feature")

    @computed_field
    def derived_features(self) -> list[str]:
        return self._get_columns_by_tag("derived_feature")

    @computed_field
    def model_features(self) -> list[str]:
        return self._get_columns_by_tag("model_feature")

    @computed_field
    def target(self) -> list[str]:
        return self._get_columns_by_tag("target")

    @computed_field
    def unique_identifier(self) -> list[str]:
        return self._get_columns_by_tag("unique_identifier")

    @computed_field
    def metadata(self) -> list[str]:
        return self._get_columns_by_tag("metadata")

    @computed_field
    def row_timestamp(self) -> list[str]:
        return self._get_columns_by_tag("row_timestamp")
