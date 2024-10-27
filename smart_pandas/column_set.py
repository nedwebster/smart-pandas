from collections import Counter
from typing import List

from pydantic import BaseModel, field_validator

from smart_pandas.column import Column


class ColumnSet(BaseModel):
    """Class to represent a set of columns."""

    columns: List[Column]

    @field_validator("columns")
    @classmethod
    def validate_column_compatibilities(cls, columns: List[Column]) -> List[Column]:
        label_collection = [column.labels.labels for column in columns]
        flattened_label_collection = [
            label for labels in label_collection for label in labels
        ]
        for label, counter in Counter(flattened_label_collection).items():
            if label.dataset_limit is not None and counter > label.dataset_limit:
                raise ValueError(
                    f"Label {label.name} occurs too many times in column set."
                )

        return columns

    def __iter__(self):
        for column in self.columns:
            yield column
