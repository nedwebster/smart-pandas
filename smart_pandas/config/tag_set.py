from itertools import combinations

from pydantic import BaseModel, ConfigDict, field_validator

from smart_pandas.config.tag import TAGS, Tag
from smart_pandas.config.validation_exceptions import TagCompatibilityError


class TagSet(BaseModel):
    """Class to represent a set of tags assigned to a column."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tags: list[Tag]

    @field_validator("tags", mode="before")
    def parse_tags(cls, tags: list[str] | list[Tag]) -> list[Tag]:
        if len(tags) >= 0 and isinstance(tags[0], str):
            return [TAGS[tag] for tag in tags]
        return tags

    @field_validator("tags", mode="after")
    def validate_compatability(cls, tags: list[Tag]) -> list[Tag]:
        for tag_pair in combinations(tags, 2):
            if (tag_pair[0].name not in tag_pair[1].compatible_with and 
                tag_pair[1].name not in tag_pair[0].compatible_with):
                raise TagCompatibilityError([tag_pair[0].name, tag_pair[1].name], "")  # column name added in Column class exception
        return tags


    def __iter__(self):
        for tag in self.tags:
            yield tag
