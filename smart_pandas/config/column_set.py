from collections import Counter
from typing import List

from pydantic import BaseModel, field_validator

from smart_pandas.config.column import Column


class ColumnSet(BaseModel):
    """Class to represent a set of columns."""

    columns: List[Column]

    @field_validator("columns")
    @classmethod
    def validate_column_compatibilities(cls, columns: List[Column]) -> List[Column]:
        tag_collection = [column.tags.tags for column in columns]
        flattened_tag_collection = [
            tag for tags in tag_collection for tag in tags
        ]
        for tag, counter in Counter(flattened_tag_collection).items():
            if tag.dataset_limit is not None and counter > tag.dataset_limit:
                raise ValueError(
                    f"Tag {tag.name} occurs too many times in column set."
                )

        return columns

    def __iter__(self):
        for column in self.columns:
            yield column
