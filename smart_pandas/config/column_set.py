from collections import Counter
from typing import List

from pydantic import BaseModel, field_validator

from smart_pandas.config.column import Column
from smart_pandas.config.validation_exceptions import (
    EmptyColumnSetError,
    DuplicateColumnError,
    TagLimitExceededError,
)


class ColumnSet(BaseModel):
    """Class to represent a set of columns."""

    columns: List[Column]

    @field_validator("columns")
    @classmethod
    def validate_column_set_length(cls, columns: List[Column]) -> List[Column]:
        if len(columns) == 0:
            raise EmptyColumnSetError()
        return columns

    @field_validator("columns")
    @classmethod
    def validate_tag_count_limits(cls, columns: List[Column]) -> List[Column]:
        tag_collection = [column.tags for column in columns]
        flattened_tag_collection = [
            tag for tags in tag_collection for tag in tags
        ]
        for tag, counter in Counter(flattened_tag_collection).items():
            if tag.dataset_limit is not None and counter > tag.dataset_limit:
                raise TagLimitExceededError(tag.name, tag.dataset_limit, counter)

        return columns

    @field_validator("columns")
    @classmethod
    def validate_unique_column_names(cls, columns: List[Column]) -> List[Column]:
        column_names = Counter([column.name for column in columns])
        duplicate_columns = [column for column, count in column_names.items() if count > 1]
        if duplicate_columns:
            raise DuplicateColumnError(duplicate_columns)
        return columns

    def __iter__(self):
        for column in self.columns:
            yield column
