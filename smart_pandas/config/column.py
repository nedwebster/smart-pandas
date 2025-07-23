from pydantic import BaseModel, ConfigDict, field_validator, model_validator, field_serializer, ValidationError
import pandera as pa

from smart_pandas.config.tag import TAGS
from smart_pandas.config.tag_set import TagSet
from smart_pandas.config.validation_exceptions import TagCompatibilityError


class Column(BaseModel):
    """Class to represent a column in a dataframe.

    Parameters
    ----------
    name: str
        The name of the column.
    data_schema: pa.Column
        The schema configuration of the column.
    tags: TagSet
        The tags associated with the column.
    description: str | None = None
        A description of the column.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    name: str
    data_schema: pa.Column
    tags: TagSet
    description: str | None = None

    @field_validator("data_schema", mode="before")
    def parse_pandera_column(cls, v):
        if isinstance(v, dict):
            return pa.Column(**v)
        return v

    @field_validator("tags", mode="before")
    def parse_tags(cls, v, values):
        if isinstance(v, list):
            try:
                return TagSet(tags=v)
            except ValidationError as e:
                error = e.errors()[0]
                if isinstance(error["ctx"].get("error"), TagCompatibilityError):
                    # re-raise TagCompatibilityError nested in the ValidationError, with the column name added
                    raise TagCompatibilityError(incompatible_tags=error["ctx"]["error"].incompatible_tags, column_name=values.data["name"])
                else:
                    raise e
        return v

    @model_validator(mode='after')
    def set_tag_attributes(self):
        """Set tag attributes on the column for easy access."""
        for tag_name in TAGS.keys():
            setattr(self, tag_name, any(tag == tag_name for tag in self.tags))
        return self

    @field_serializer("data_schema")
    def serialize_data_schema(self, data_schema: pa.Column) -> dict:
        dict_schema = data_schema.properties
        dict_schema["dtype"] = dict_schema["dtype"].type.name
        return dict_schema
