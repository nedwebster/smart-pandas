from typing import List
from itertools import combinations

from pydantic import BaseModel, field_validator

from smart_pandas.label import Label


class LabelSet(BaseModel):
    """Class to represent a set of labels assigned to a column."""

    labels: List[Label]

    @field_validator("labels")
    @classmethod
    def validate_compatability(cls, labels: List[Label]) -> List[Label]:
        for label_pair in combinations(labels, 2):
            if label_pair[0] not in label_pair[1].compatible_with:
                raise ValueError(
                    f"Labels {label_pair[0].name} and {label_pair[1].name} are not compatible"
                )
        return labels

    def __iter__(self):
        for label in self.labels:
            yield label
