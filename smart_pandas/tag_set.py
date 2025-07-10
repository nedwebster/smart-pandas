from typing import List
from itertools import combinations

from pydantic import BaseModel, field_validator

from smart_pandas.tag import Tag


class TagSet(BaseModel):
    """Class to represent a set of tags assigned to a column."""

    tags: List[Tag]

    @field_validator("tags")
    @classmethod
    def validate_compatability(cls, tags: List[Tag]) -> List[Tag]:
        for tag_pair in combinations(tags, 2):
            if tag_pair[0] not in tag_pair[1].compatible_with:
                raise ValueError(
                    f"Tags {tag_pair[0].name} and {tag_pair[1].name} are not compatible"
                )
        return tags

    def __iter__(self):
        for tag in self.tags:
            yield tag 