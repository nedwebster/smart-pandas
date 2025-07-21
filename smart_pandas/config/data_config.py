from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from smart_pandas.config.column_set import ColumnSet
from smart_pandas.config.tag import TAGS

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

    @model_validator(mode="after")
    def set_data_attributes(self):
        """Set the data attributes dynamically based on the column set."""
        for tag in TAGS.values():
            setattr(self, tag.data_attribute_name, self._get_columns_by_tag(tag.name))
        return self

    def _get_columns_by_tag(self, tag_name: str) -> list[str]:
        """Helper method to get column names by tag."""
        return [
            column.name for column in self.columns
            if tag_name in column.tags
        ]
