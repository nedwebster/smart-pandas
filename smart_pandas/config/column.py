from pydantic import BaseModel, ConfigDict, field_validator, model_validator, field_serializer, ValidationError
import pandera as pa

from smart_pandas.config.tag import TAGS
from smart_pandas.config.tag_set import TagSet
from smart_pandas.config.validation_exceptions import TagCompatibilityError
from smart_pandas.state import StateName


_STATE_NAMES = {s.value for s in StateName}


class StateDependentSchema(BaseModel):
    """A mapping of state names to Pandera column schemas.

    Used when a column's data type varies by state (e.g. str in RAW, float in PROCESSED).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    schemas: dict[str, pa.Column]

    @field_validator("schemas", mode="before")
    def validate_state_keys(cls, v):
        invalid_keys = set(v.keys()) - _STATE_NAMES
        if invalid_keys:
            raise ValueError(
                f"Invalid state name(s) in data_schema: {invalid_keys}. "
                f"Valid state names are: {_STATE_NAMES}"
            )
        return v


class Column(BaseModel):
    """Class to represent a column in a dataframe.

    Parameters
    ----------
    name: str
        The name of the column.
    data_schema: StateDependentSchema | pa.Column
        The schema configuration of the column. Can be a single Pandera column
        schema applied in all states, or a StateDependentSchema mapping state
        names (e.g. "raw", "processed") to different schemas per state.
    tags: TagSet
        The tags associated with the column.
    description: str | None = None
        A description of the column.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    name: str
    data_schema: StateDependentSchema | pa.Column
    tags: TagSet
    description: str | None = None

    @field_validator("data_schema", mode="before")
    def parse_pandera_column(cls, v):
        if isinstance(v, dict):
            # If keys are state names, parse as state-dependent schema
            if v.keys() & _STATE_NAMES:
                schemas = {
                    state_key: pa.Column(**schema) if isinstance(schema, dict) else schema
                    for state_key, schema in v.items()
                }
                return StateDependentSchema(schemas=schemas)
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

    def get_data_schema(self, state_name: StateName) -> pa.Column:
        """Get the data schema for a given state.

        Parameters
        ----------
        state_name : StateName
            The current state name to resolve the schema for.

        Returns
        -------
        pa.Column
            The resolved Pandera column schema.
        """
        if isinstance(self.data_schema, StateDependentSchema):
            if state_name.value in self.data_schema.schemas:
                return self.data_schema.schemas[state_name.value]
            # Fall back to the first defined schema
            return next(iter(self.data_schema.schemas.values()))
        return self.data_schema

    @field_serializer("data_schema")
    def serialize_data_schema(self, data_schema: pa.Column | StateDependentSchema) -> dict:
        if isinstance(data_schema, StateDependentSchema):
            result = {}
            for state_key, schema in data_schema.schemas.items():
                schema_dict = schema.properties.copy()
                schema_dict["dtype"] = schema_dict["dtype"].type.name
                result[state_key] = schema_dict
            return result
        dict_schema = data_schema.properties
        dict_schema["dtype"] = dict_schema["dtype"].type.name
        return dict_schema
