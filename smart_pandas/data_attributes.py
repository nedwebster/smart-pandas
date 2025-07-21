from pydantic import BaseModel
from smart_pandas.state import StateName, MLStage


class DataAttributes(BaseModel):
    """Data attributes of the SmartPandas accessor."""
    name: str
    tag_name: str
    invalid_states: list[StateName] = []  # TODO: Move state compatibility to State class
    invalid_ml_stages: list[MLStage] = []

DATA_ATTRIBUTES = [
    DataAttributes(name="raw_features", tag_name="raw_feature"),
    DataAttributes(name="derived_features", tag_name="derived_feature", invalid_states=[StateName.RAW]),
    DataAttributes(name="model_features", tag_name="model_feature", invalid_states=[StateName.RAW]),
    DataAttributes(name="target", tag_name="target", invalid_ml_stages=[MLStage.INFERENCE]),
    DataAttributes(name="unique_identifier", tag_name="unique_identifier"),
    DataAttributes(name="metadata", tag_name="metadata"),
    DataAttributes(name="row_timestamp", tag_name="row_timestamp"),
]
